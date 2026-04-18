import requests
import os
import time
import random

# --- CONFIGURATION ---
SERVER_URL = "http://127.0.0.1:5000/detect_hazards"
# Pointing to your specific Atrium3 North folder
IMAGE_DIR = r"C:\Users\anguy\OneDrive\Desktop\Indoor-MR-Assist-LV\FastVLM_Server\FastVLM_Yolo_IPS\atrium3_north_split8_FOV90_P3"

def run_random_batch_test(count=5):
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Folder not found at {IMAGE_DIR}")
        return

    # 1. Gather all valid image files
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    if len(all_files) < count:
        print(f"Warning: Only {len(all_files)} images found. Testing all of them.")
        test_files = all_files
    else:
        # 2. Randomly select 5 unique images
        test_files = random.sample(all_files, count)

    print(f"[*] Starting Randomized Test: {len(test_files)} images selected from {len(all_files)} total.")
    print("-" * 55)

    for i, img_name in enumerate(test_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # Consistent mock data for testing logic across different Atrium angles
        payload = {
            'prompt': f'Random Test {i+1}',
            'center_depth': '4.0',  # 4 meters (Atrium spaces are large)
            'real_height': '1.6',   # 1.6 meters user height
            'head_pitch': '0.0'     # Looking straight ahead
        }

        try:
            print(f"[{i+1}/{count}] Processing: {img_name}...")
            
            with open(img_path, 'rb') as img:
                files = {'image': img}
                # YOLO usually takes < 1s, but we give it 30s for safety
                response = requests.post(SERVER_URL, data=payload, files=files, timeout=120)

            if response.status_code == 200:
                result = response.json()
                print(f"    Server Reported Latency: {result.get('latency')}")
                print(f"    Final Speech Output:    {result.get('final_response')}")
                print("-" * 35)
            else:
                print(f"    [!] Server Error: Status {response.status_code}")

        except Exception as e:
            print(f"    [!] Connection/File Error: {e}")

        # 3. Give the Surface Pro a breather (Sequential processing safety)
        time.sleep(2)

    print("\n[FINISH] Random batch test complete. Check your server's debug_captures folder.")

if __name__ == "__main__":
    run_random_batch_test(count=5)