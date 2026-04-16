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
YOLO_PATH = r"C:\Users\anguy\OneDrive\Desktop\Indoor-MR-Assist-LV\FastVLM_Server\FastVLM_Yolo_IPS\yolo26x.pt"
DEBUG_SAVE_DIR = r"C:\Users\anguy\OneDrive\Desktop\Indoor-MR-Assist-LV\FastVLM_Server\FastVLM_Yolo_IPS\debug_captures"
HOST_IP = '0.0.0.0'
PORT = 5000

# Constants
M_TO_FT = 3.28084
MAX_FT = 40.0

if not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR)

print(f"[*] Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_PATH)
yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')

def get_direction(norm_x):
    if norm_x < 0.35: return "on your left"
    elif norm_x > 0.65: return "on your right"
    else: return "ahead of you"

def add_article(word):
    return f"an {word}" if word.lower().startswith(('a','e','i','o','u')) else f"a {word}"

def fuse_distance_feet(ny_bottom, quest_depth_m, real_height_m, head_pitch_deg, label):
    q_depth_m = float(quest_depth_m)
    h = float(real_height_m)
  

    # 1. VALIDATE HARDWARE DEPTH
    # Quest 3 is accurate between 0.2m and 5m indoors.
    # If the object is near the middle of the screen (ny_bottom between 0.3 and 0.7),
    # the center_depth sensor is hitting it directly.
    if 0.2 < q_depth_m < 8.0:
        # If the object is roughly in the center vertical third, trust hardware 100%
        if 0.3 < ny_bottom < 0.7:
            final_m = q_depth_m
        else:
            # For objects at the floor or ceiling, blend with Trig to account for perspective
            pitch_rad = np.radians(float(head_pitch_deg))
            v_fov_rad = np.radians(60)
            pixel_angle_rad = (0.5 - ny_bottom) * v_fov_rad
            total_angle_rad = pitch_rad + pixel_angle_rad
          

            denom = np.tan(total_angle_rad)
            trig_dist_m = h / denom if denom > 0.1 else q_depth_m
           

            # Weighted blend: Still favor hardware depth heavily
            final_m = (q_depth_m * 0.8) + (trig_dist_m * 0.2)
    else:
        # Fallback to pure Trig if sensor is out of range
        final_m = 3.0 # Default safe distance for navigation

    return round(final_m * 3.28084, 1)

def format_group(items_with_data, direction):
    if not items_with_data: return None
    sorted_items = sorted(items_with_data, key=lambda x: x[1])
    seen = set()
    top_items = []
    for label, dist in sorted_items:
        if label not in seen:
            top_items.append(f"{add_article(label)} at {dist} feet")
            seen.add(label)
        if len(top_items) == 3: break

    if not top_items: return None
    if len(top_items) == 1: item_string = top_items[0]
    elif len(top_items) == 2: item_string = f"{top_items[0]} and {top_items[1]}"
    else: item_string = f"{top_items[0]}, {top_items[1]}, and {top_items[2]}"
    return f"{item_string} {direction}"

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    start_time = time.time()
    img_file = request.files.get('image')
    user_input = request.form.get('prompt', 'Voice Triggered')
    center_dist = request.form.get('center_depth', '0.0')
    real_h = request.form.get('real_height', '1.6')
    head_pitch = request.form.get('head_pitch', '0.0')

    if not img_file: return jsonify({"status": "error"}), 400

    img_bytes = img_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if cv_img is None: return jsonify({"status": "error"}), 400
   

    # Save Raw for VLM processing later
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"raw_{timestamp}.jpg"), cv_img)

    img_height, img_width = cv_img.shape[:2]
    results = yolo_model.predict(cv_img, conf=0.40, verbose=False)
   

    groups = {"ahead of you": [], "on your left": [], "on your right": []}
    raw_list = []
    debug_img = cv_img.copy()

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            nx = float(((bbox[0] + bbox[2]) / 2) / img_width)
            ny_bottom = float(bbox[3] / img_height)
           

            # THE CRITICAL CALL
            final_dist_ft = fuse_distance_feet(ny_bottom, center_dist, real_h, head_pitch, label)

           
            dir_key = get_direction(nx)
            groups[dir_key].append((label, final_dist_ft))
            raw_list.append(f"{label} ({dir_key}, {final_dist_ft}ft)")

            # Visual Debugging
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{label} {final_dist_ft}ft", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"debug_{timestamp}.jpg"), debug_img)

    speech_parts = [format_group(groups[d], d) for d in ["ahead of you", "on your left", "on your right"]]
    speech_parts = [p for p in speech_parts if p]
    final_response = "I see " + ". Also, ".join(speech_parts) if speech_parts else "Path appears clear."

    latency_ms = (time.time() - start_time) * 1000

    print("\n" + "-" * 45)
    print(f"User Input:    '{user_input}'")  
    print(f"Latency:        {latency_ms:.0f}ms")
    print(f"H_In: {real_h}m | Pitch: {head_pitch}deg")
    print(f"Quest Sensor says: {float(center_dist) * 3.28:.2f} ft")
    print(f"Yolo Raw:       {', '.join(raw_list) if raw_list else 'None'}")
    print(f"Final Response: {final_response}")
    print("-" * 45)

    return jsonify({"status": "success", "final_response": final_response, "latency": f"{latency_ms:.0f}ms"})

if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, debug=False, threaded=True) 