import sys, os, gc, io, time, threading, cv2, datetime
import numpy as np
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
from transformers import TextStreamer

# LLaVA / VLM Specific Imports
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
YOLO_PATH = str(BASE_DIR / "yolov8s-worldv2.pt")
# SWITCHED TO 0.5B MODEL
VLM_PATH = str(BASE_DIR / "llava-fastvithd_0.5b_stage2")
DEBUG_SAVE_DIR = str(BASE_DIR / "debug_captures")

if not os.path.exists(DEBUG_SAVE_DIR): os.makedirs(DEBUG_SAVE_DIR)

DEVICE = "cpu"
TORCH_DTYPE = torch.bfloat16
M_TO_FT = 3.28084
model_lock = threading.Lock()

# --- MODEL LOADING ---
def load_engines():
    global tokenizer, vlm_model, image_processor, yolo_model
    disable_torch_init()
    
    print(f"[*] Loading YOLO-World...")
    yolo_model = YOLO(YOLO_PATH)
    nav_classes = ["sign", "door", "window", "sofa", "couch", "table", "chair", "television", "refrigerator", "person", "backpack", "suitcase"]
    yolo_model.set_classes(nav_classes)

    print(f"[*] Loading FastVLM 0.5B (Stage 2)...")
    model_name = get_model_name_from_path(VLM_PATH)
    tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        VLM_PATH, 
        None, 
        model_name, 
        device=DEVICE, 
        torch_dtype=TORCH_DTYPE
    )
    vlm_model.eval()
    print(f"\n✓ SYSTEMS READY ON {DEVICE.upper()}\n")

# --- UTILITIES ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def fuse_distance_feet(ny_bottom, quest_depth_m, real_height_m, head_pitch_deg):
    q_depth_m = float(quest_depth_m); h = float(real_height_m)
    pitch_rad = np.radians(float(head_pitch_deg))
    v_fov_rad = np.radians(65) 
    total_angle_rad = pitch_rad + ((0.5 - ny_bottom) * v_fov_rad)
    denom = np.tan(total_angle_rad)
    trig_dist_m = abs(h / denom) if denom > 0.05 else 8.0
    final_m = (q_depth_m * 0.95) + (trig_dist_m * 0.05) if 0.1 < q_depth_m < 8.0 else trig_dist_m
    return round(final_m * M_TO_FT, 1)

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    if model_lock.locked(): return jsonify({"status": "busy"}), 429

    with model_lock:
        start_time = time.time()
        img_file = request.files.get('image')
        center_dist = request.form.get('center_depth', '0.0')
        real_h = request.form.get('real_height', '1.6')
        head_pitch = request.form.get('head_pitch', '0.0')

        if not img_file: return jsonify({"status": "error"}), 400

        # 1. DECODE & SAVE RAW
        nparr = np.frombuffer(img_file.read(), np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w = cv_img.shape[:2]
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"raw_{ts}.jpg"), cv_img)

        # 2. YOLO PROCESSING
        results = yolo_model.predict(cv_img, conf=0.02, verbose=False)
        temp_detections = []
        for r in results:
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                if conf < 0.15: continue
                temp_detections.append({'label': yolo_model.names[int(box.cls[0])], 'bbox': bbox, 'conf': conf})

        # NMS Filtering
        final_detections = []
        temp_detections = sorted(temp_detections, key=lambda x: x['conf'], reverse=True)
        for det in temp_detections:
            if all(calculate_iou(det['bbox'], f['bbox']) < 0.40 for f in final_detections):
                final_detections.append(det)

        # Save YOLO Box Debug
        debug_img = cv_img.copy()
        raw_list = []
        for det in final_detections:
            dist = fuse_distance_feet(det['bbox'][3]/img_h, center_dist, real_h, head_pitch)
            cv2.rectangle(debug_img, (int(det['bbox'][0]), int(det['bbox'][1])), (int(det['bbox'][2]), int(det['bbox'][3])), (0, 255, 0), 2)
            raw_list.append(f"{det['label']} at {dist}ft")
        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"yolo_obj_box_{ts}.jpg"), debug_img)

        # 3. FASTVLM 0.5B PROCESSING
        vlm_res = "No specific details."
        if final_detections:
            # Crop around top detection
            best = final_detections[0]['bbox']
            cx, cy = int((best[0]+best[2])/2), int((best[1]+best[3])/2)
            cs = 224
            x1, y1 = max(0, cx-cs//2), max(0, cy-cs//2)
            crop = cv_img[y1:min(img_h, y1+cs), x1:min(img_w, x1+cs)]
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"fastvlm_cropped_{ts}.jpg"), crop)

            # VLM Inference
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_tensor = process_images([pil_img], image_processor, vlm_model.config)[0].unsqueeze(0).to(DEVICE, dtype=TORCH_DTYPE)
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nIdentify this object concisely.")
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)

            with torch.inference_mode():
                output_ids = vlm_model.generate(input_ids, images=image_tensor, image_sizes=[pil_img.size], do_sample=False, max_new_tokens=12)
            vlm_res = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 4. FINAL LOGGING
        latency_ms = (time.time() - start_time) * 1000
        yolo_str = ", ".join(raw_list) if raw_list else "None"
        final_res = f"I see {yolo_str}. Detailed as: {vlm_res}"

        print("\n" + "-" * 45)
        print(f"YOLO Processing: Found {len(final_detections)} objects.")
        print(f"FastVLM (0.5B): {vlm_res}")
        print(f"Combined Speech: {final_res}")
        print(f"Total Latency:   {latency_ms:.0f}ms")
        print("-" * 45)

        return jsonify({"status": "success", "final_response": final_res, "latency": f"{latency_ms:.0f}ms"})

if __name__ == '__main__':
    load_engines()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)