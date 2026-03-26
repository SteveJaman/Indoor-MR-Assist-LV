import io
import time
import threading
import requests
import imageio.v3 as iio
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# --- CONFIG ---
JETSON_IP = "10.0.0.172"
JETSON_URL = f"http://{JETSON_IP}:5000/detect_hazards"

# Path Logic: Points to the /samples subfolder
BASE_DIR = Path(__file__).parent
SAMPLE_DIR = BASE_DIR / "samples"
SAVE_DIR = BASE_DIR / "ai_snapshots"

# Ensure directories exist
SAMPLE_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True) 

# --- SELECTION POOLS ---
IMAGE_POOL = {
    "1": "car.jpg", 
    "2": "2026 Jan. FUWAMOCO Doggy Pack Wallpaper.png"
}
VIDEO_POOL = {
    "1": "deco.mp4",
    "2": "TimeGhost.mp4" 
}

# Global flag for Adaptive Logic
jetson_ready = True 

def ai_worker(image_frame, frame_id, video_name):
    """Background task: Sends frame to Jetson and saves documentation."""
    global jetson_ready
    try:
        timestamp = int(time.time())
        snap_path = SAVE_DIR / f"{video_name}_frame_{frame_id}_{timestamp}.jpg"
        image_frame.save(snap_path)

        print(f"\n[AI START] Analyzing {video_name} (Frame {frame_id})...")
        
        buffer = io.BytesIO()
        image_frame.save(buffer, format='JPEG', quality=75) 
        buffer.seek(0)
        
        start = time.time()
        response = requests.post(JETSON_URL, files={'image': buffer}, timeout=60)
        
        if response.status_code == 200:
            res = response.json().get('ai_response')
            print(f"\n[JETSON ANALYSIS - {video_name}] ({time.time()-start:.2f}s):")
            print(f"{'='*40}\n{res}\n{'='*40}")
    except Exception as e:
        print(f"\n[AI ERROR]: {e}")
    finally:
        jetson_ready = True

def run_image_test():
    print("\n--- Image Pool (Source: /samples) ---")
    for k, v in IMAGE_POOL.items(): print(f"[{k}] {v}")
    idx = input("Select Image # > ")
    if idx in IMAGE_POOL:
        img_path = SAMPLE_DIR / IMAGE_POOL[idx]
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            print(f"[*] Sending {IMAGE_POOL[idx]}...")
            response = requests.post(JETSON_URL, files={'image': buffer})
            print(f"\n[RESULT]: {response.json().get('ai_response')}")
        else:
            print(f"[!] {IMAGE_POOL[idx]} not found in {SAMPLE_DIR}")

def run_continuous_video(filename):
    global jetson_ready
    vid_path = SAMPLE_DIR / filename
    if not vid_path.exists():
        print(f"[!] {filename} not found in {SAMPLE_DIR}")
        return

    print(f"[*] Analyzing Stream: {filename}")
    
    plt.ion()
    fig, ax = plt.subplots()
    im_display = None
    jetson_ready = True 

    try:
        for idx, frame in enumerate(iio.imiter(vid_path, plugin="pyav")):
            if im_display is None:
                im_display = ax.imshow(frame)
            else:
                im_display.set_data(frame)
            
            status = "READY" if jetson_ready else "THINKING"
            plt.title(f"Video: {filename} | Frame: {idx} | Jetson: {status}")
            plt.draw()
            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                break

            if jetson_ready:
                jetson_ready = False 
                current_pil = Image.fromarray(frame).resize((448, 448))
                clean_name = Path(filename).stem
                thread = threading.Thread(target=ai_worker, args=(current_pil, idx, clean_name))
                thread.daemon = True
                thread.start()
    finally:
        plt.close(fig)
        plt.ioff()

def main_menu():
    while True:
        print("\n=== VLM RESEARCH SUITE (SAMPLES MODE) ===")
        print("[1] Test Static Images")
        print("[2] Run Continuous Video (Adaptive)")
        print("[Q] Exit")
        
        choice = input("Selection > ").strip().lower()

        if choice == '1':
            run_image_test()
        elif choice == '2':
            print("\n-- Video Pool (Source: /samples) --")
            for k, v in VIDEO_POOL.items(): print(f"[{k}] {v}")
            idx = input("Select Video # > ")
            if idx in VIDEO_POOL:
                run_continuous_video(VIDEO_POOL[idx])
        elif choice == 'q':
            break

if __name__ == "__main__":
    main_menu()
