import sys, os, gc, io, time, threading, cv2, datetime
import numpy as np
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

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
VLM_PATH = str(BASE_DIR / "llava-fastvithd_0.5b_stage2")
DEBUG_SAVE_DIR = str(BASE_DIR / "debug_captures")

if not os.path.exists(DEBUG_SAVE_DIR): os.makedirs(DEBUG_SAVE_DIR)

DEVICE = "cpu"
TORCH_DTYPE = torch.bfloat16
M_TO_FT = 3.28084
model_lock = threading.Lock()

def load_engines():
    global tokenizer, vlm_model, image_processor, yolo_model
    disable_torch_init()
    
    print(f"[*] Loading YOLO-World...")
    yolo_model = YOLO(YOLO_PATH)
    yolo_model.set_classes(["sign", "door", "window", "sofa", "couch", "table", "chair", "television", "refrigerator", "person", "backpack", "suitcase"])

    print(f"[*] Loading FastVLM 0.5B (Stage 2)...")
    model_name = get_model_name_from_path(VLM_PATH)
    tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        VLM_PATH, None, model_name, device=DEVICE, torch_dtype=TORCH_DTYPE
    )
    vlm_model.eval()
    print(f"\n✓ SYSTEMS READY ON {DEVICE.upper()}\n")

# --- HELPER FUNCTIONS ---
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def get_direction(norm_x):
    if norm_x < 0.35: return "on your left"
    elif norm_x > 0.65: return "on your right"
    else: return "ahead of you"

def fuse_distance_feet(ny_bottom, quest_depth_m, real_height_m, head_pitch_deg):
    q_depth_m = float(quest_depth_m)
    h = float(real_height_m)
    pitch_deg = float(head_pitch_deg)
    if pitch_deg > 15 and ny_bottom > 0.6: h = h * 0.6 
    pitch_rad = np.radians(pitch_deg)
    v_fov_rad = np.radians(65) 
    total_angle_rad = pitch_rad + ((0.5 - ny_bottom) * v_fov_rad)
    denom = np.tan(total_angle_rad)
    trig_dist_m = abs(h / denom) if denom > 0.05 else 8.0
    if 0.1 < q_depth_m < 8.0: final_m = (q_depth_m * 0.95) + (trig_dist_m * 0.05)
    else: final_m = trig_dist_m
    return round(final_m * M_TO_FT, 1)

def format_group(items_with_data, direction):
    if not items_with_data: return None
    seen, top_items = set(), []
    for label, dist in sorted(items_with_data, key=lambda x: x[1]):
        if label not in seen:
            article = "an" if label.lower().startswith(('a','e','i','o','u')) else "a"
            top_items.append(f"{article} {label} at {dist} feet")
            seen.add(label)
        if len(top_items) == 3: break
    return f"{', '.join(top_items)} {direction}"

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    start_req_time = time.time()
    img_file = request.files.get('image')
    if not img_file: return jsonify({"status": "error"}), 400

    # 1. IMMEDIATE DEBUG SAVE
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nparr = np.frombuffer(img_file.read(), np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{ts}_1_raw.jpg"), cv_img)

    # 2. LOCK MANAGEMENT
    acquired = model_lock.acquire(timeout=15)
    if not acquired:
        print(f"[!] DROPPED: {ts} (Server Busy)")
        return jsonify({"status": "busy"}), 429

    try:
        img_h, img_w = cv_img.shape[:2]
        center_dist = request.form.get('center_depth', '0.0')
        real_h = request.form.get('real_height', '1.6')
        head_pitch = request.form.get('head_pitch', '0.0')

        # 3. YOLO PROCESSING
        yolo_start = time.time()
        results = yolo_model.predict(cv_img, conf=0.15, verbose=False)
        temp_dets = []
        for r in results:
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                temp_dets.append({
                    'label': yolo_model.names[int(box.cls[0])], 
                    'bbox': bbox, 'conf': float(box.conf[0]), 
                    'nx': float(((bbox[0]+bbox[2])/2)/img_w), 
                    'nyb': float(bbox[3]/img_h)
                })

        final_detections = []
        temp_dets = sorted(temp_dets, key=lambda x: x['conf'], reverse=True)
        groups = {"ahead of you": [], "on your left": [], "on your right": []}
        debug_viz = cv_img.copy()

        for det in temp_dets:
            if all(calculate_iou(det['bbox'], f['bbox']) < 0.40 for f in final_detections):
                final_detections.append(det)
                dist = fuse_distance_feet(det['nyb'], center_dist, real_h, head_pitch)
                groups[get_direction(det['nx'])].append((det['label'], dist))
                
                # Draw on Debug Viz
                b = det['bbox'].astype(int)
                cv2.rectangle(debug_viz, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(debug_viz, f"{det['label']} {dist}ft", (b[0], b[1]-10), 0, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{ts}_2_yolo_viz.jpg"), debug_viz)
        yolo_time = (time.time() - yolo_start) * 1000

        # 4. VLM PROCESSING
        vlm_start = time.time()
        vlm_res = "No objects for VLM."
        if final_detections:
            best = final_detections[0]['bbox']
            cx, cy = int((best[0]+best[2])/2), int((best[1]+best[3])/2)
            y1, y2, x1, x2 = max(0, cy-112), min(img_h, cy+112), max(0, cx-112), min(img_w, cx+112)
            crop = cv_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{ts}_3_vlm_crop.jpg"), crop)
            
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_tensor = process_images([pil_img], image_processor, vlm_model.config)[0].unsqueeze(0).to(DEVICE, dtype=TORCH_DTYPE)
            
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe this object concisely.")
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                output_ids = vlm_model.generate(input_ids, images=image_tensor, image_sizes=[pil_img.size], do_sample=False, max_new_tokens=15)
            vlm_res = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        vlm_time = (time.time() - vlm_start) * 1000

        # 5. RESPONSE CONSTRUCTION
        speech_parts = [format_group(groups[d], d) for d in ["ahead of you", "on your left", "on your right"]]
        yolo_text = "I see " + ". Also, ".join([p for p in speech_parts if p]) if any(speech_parts) else "Path clear."
        final_res = f"{yolo_text} {vlm_res}"
        total_latency = (time.time() - start_req_time) * 1000

        # --- DEBUG LOGGER BLOCK ---
        print("\n" + "="*50)
        print(f"ID: {ts}")
        print(f"LATENCY:  Total: {total_latency:.0f}ms | YOLO: {yolo_time:.0f}ms | VLM: {vlm_time:.0f}ms")
        print(f"OBJECTS:  {', '.join([d['label'] for d in final_detections]) if final_detections else 'None'}")
        print(f"VLM DESC: {vlm_res}")
        print(f"REPLY:    {final_res}")
        print("="*50)

        return jsonify({"status": "success", "final_response": final_res, "latency": f"{total_latency:.0f}ms"})

    except Exception as e:
        print(f"[!] ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        model_lock.release()
        gc.collect()

if __name__ == '__main__':
    load_engines()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)