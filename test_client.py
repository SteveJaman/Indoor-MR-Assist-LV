import requests
import os

# Configuration
SERVER_URL = "http://127.0.0.1:5000/detect_hazards"
IMAGE_PATH = "frame_004_cam_8.jpg" 

def send_test_request():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # 1. Cleaned up payload
    # Note: Your current Flask server only looks for the 'image' file.
    # Extra keys like 'prompt' in 'data' are ignored by the server but kept here for future-proofing.
    data = {'prompt': 'Describe the path.'}

    try:
        print(f"[*] Sending 1.5B VLM Request to {SERVER_URL}...")
        print("[*] Note: The server progress bar may stay at 0% for several minutes during 'Prefill'.")
        
        with open(IMAGE_PATH, 'rb') as img:
            files = {'image': img}
            
            # Using a tuple for (connect_timeout, read_timeout)
            # 10s to connect, 600s to wait for the heavy 1.5B math to finish.
            response = requests.post(SERVER_URL, data=data, files=files, timeout=(10, 600))

        if response.status_code == 200:
            print("\n[SUCCESS] Server Responded:")
            result = response.json()
            print(f"Response: {result.get('final_response')}")
            print(f"Latency: {result.get('latency')}ms")
        else:
            print(f"\n[FAILED] Status Code: {response.status_code}")
            print(response.text)

    except requests.exceptions.Timeout:
        print("\n[ERROR] Request Timed Out. The 1.5B model is taking longer than 10 minutes to prefill.")
    except Exception as e:
        print(f"\n[ERROR] Connection Error: {e}")

if __name__ == "__main__":
    send_test_request()