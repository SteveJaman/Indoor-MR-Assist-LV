from flask import Flask, request, jsonify
import datetime
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import time

app = Flask(__name__)

# --- CONFIGURATION ---
YOLO_PATH = r"C:\Users\anguy\OneDrive\Desktop\Indoor-MR-Assist-LV\FastVLM_Server\FastVLM_Yolo_IPS\yolov8s-worldv2.pt"
DEBUG_SAVE_DIR = r"C:\Users\anguy\OneDrive\Desktop\Indoor-MR-Assist-LV\FastVLM_Server\FastVLM_Yolo_IPS\debug_captures"
HOST_IP = '0.0.0.0'
PORT = 5000
M_TO_FT = 3.28084

if not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR)

print(f"[*] Loading YOLO-World (v8: Distance Correction & High-Trust Depth)...")
yolo_model = YOLO(YOLO_PATH)

nav_classes = [
    "sign", "exit sign", "door", "window",
    "sofa", "couch", "coffee table", "dining table", "chair", "television", 
    "refrigerator", "microwave", "bed", "toilet", "sink",
    "red fire extinguisher", "fire alarm",
    "walking person", "human silhouette",
    "trash can", "backpack", "suitcase",
    "ribbon decoration", "wall outlet", "wall panel", "ceiling light"
]
yolo_model.set_classes(nav_classes)
yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def get_direction(norm_x):
    if norm_x < 0.35: return "on your left"
    elif norm_x > 0.65: return "on your right"
    else: return "ahead of you"

def add_article(word):
    return f"an {word}" if word.lower().startswith(('a','e','i','o','u')) else f"a {word}"

def fuse_distance_feet(ny_bottom, quest_depth_m, real_height_m, head_pitch_deg):
    q_depth_m = float(quest_depth_m)
    h = float(real_height_m)
    pitch_deg = float(head_pitch_deg)
    
    # Adjust height if the user is likely sitting (looking down at low furniture)
    if pitch_deg > 15 and ny_bottom > 0.6:
        h = h * 0.6 # Assume 60% of standing height
    
    pitch_rad = np.radians(pitch_deg)
    v_fov_rad = np.radians(65) 
    pixel_angle_rad = (0.5 - ny_bottom) * v_fov_rad
    total_angle_rad = pitch_rad + pixel_angle_rad
    
    denom = np.tan(total_angle_rad)
    trig_dist_m = abs(h / denom) if denom > 0.05 else 8.0
    
    # NEW WEIGHTING: Trust Quest sensor almost exclusively if it has a reading
    # This prevents the 'double distance' trig error
    if 0.1 < q_depth_m < 8.0:
        final_m = (q_depth_m * 0.95) + (trig_dist_m * 0.05)
    else:
        final_m = trig_dist_m
        
    return round(final_m * M_TO_FT, 1)

def format_group(items_with_data, direction):
    if not items_with_data: return None
    sorted_items = sorted(items_with_data, key=lambda x: x[1])
    seen, top_items = set(), []
    for label, dist in sorted_items:
        if label not in seen:
            top_items.append(f"{add_article(label)} at {dist} feet")
            seen.add(label)
        if len(top_items) == 4: break
    if not top_items: return None
    if len(top_items) == 1: item_string = top_items[0]
    else: item_string = ", ".join(top_items[:-1]) + f", and {top_items[-1]}"
    return f"{item_string} {direction}"

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    start_time = time.time()
    img_file = request.files.get('image')
    user_input = request.form.get('prompt', 'Voice')
    center_dist = request.form.get('center_depth', '0.0')
    real_h = request.form.get('real_height', '1.6')
    head_pitch = request.form.get('head_pitch', '0.0')

    if not img_file: return jsonify({"status": "error"}), 400
    nparr = np.frombuffer(img_file.read(), np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_h, img_w = cv_img.shape[:2]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"raw_{timestamp}.jpg"), cv_img)
    
    results = yolo_model.predict(cv_img, conf=0.02, verbose=False)

    temp_detections = []
    for r in results:
        for box in r.boxes:
            raw_label = yolo_model.names[int(box.cls[0])]
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            ny_bottom = float(bbox[3]/img_h)
            nx = float(((bbox[0]+bbox[2])/2)/img_w)
            
            if any(x in raw_label for x in ["ribbon", "outlet", "panel", "light"]):
                continue

            # Home Furniture mapping
            if any(x in raw_label for x in ["sofa", "couch", "table", "chair", "tv", "bed", "toilet", "sink", "fridge"]):
                if conf < 0.18: continue
                label = "table" if "table" in raw_label else raw_label
                if "television" in raw_label: label = "TV"
            elif "sign" in raw_label:
                if conf < 0.15: continue
                label = "sign"
            elif any(x in raw_label for x in ["door", "window"]):
                if conf < 0.08: continue 
                label = raw_label
            elif "person" in raw_label:
                if conf < 0.30: continue 
                label = "person"
            else:
                continue

            temp_detections.append({'label': label, 'bbox': bbox, 'conf': conf, 'nx': nx, 'nyb': ny_bottom})

    final_detections = []
    temp_detections = sorted(temp_detections, key=lambda x: x['conf'], reverse=True)
    for det in temp_detections:
        keep = True
        for existing in final_detections:
            if calculate_iou(det['bbox'], existing['bbox']) > 0.40:
                keep = False; break
        if keep: final_detections.append(det)

    groups = {"ahead of you": [], "on your left": [], "on your right": []}
    raw_list, debug_img = [], cv_img.copy()

    for det in final_detections:
        dist = fuse_distance_feet(det['nyb'], center_dist, real_h, head_pitch)
        dir_k = get_direction(det['nx'])
        groups[dir_k].append((det['label'], dist))
        
        b = [round(float(x), 1) for x in det['bbox']]
        raw_list.append(f"{det['label']} {b}")
        
        cv2.rectangle(debug_img, (int(det['bbox'][0]), int(det['bbox'][1])), (int(det['bbox'][2]), int(det['bbox'][3])), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{det['label']} {dist}ft", (int(det['bbox'][0]), int(det['bbox'][1])-10), 0, 0.5, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"debug_{timestamp}.jpg"), debug_img)
    
    speech = [format_group(groups[d], d) for d in ["ahead of you", "on your left", "on your right"]]
    speech = [p for p in speech if p]
    final_res = "I see " + ". Also, ".join(speech) if speech else "Path appears clear."

    latency_ms = (time.time() - start_time) * 1000

    # --- DEBUG LOGGING ---
    print("\n" + "-" * 45)
    print(f"User Input:     '{user_input}'")  
    print(f"Latency:        {latency_ms:.0f}ms")
    print(f"Yolo Raw:       {', '.join(raw_list) if raw_list else 'None'}")
    print(f"Final Response: {final_res}")
    print("-" * 45)

    return jsonify({"status": "success", "final_response": final_res, "latency": f"{latency_ms:.0f}ms"})

if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, debug=False, threaded=True)