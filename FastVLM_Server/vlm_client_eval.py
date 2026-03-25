import requests
import os
import time

# --- CONFIG ---
JETSON_IP = "10.0.0.172" 
URL = f"http://{JETSON_IP}:5000/detect_hazards"
IMAGE_NAME = "car.jpg"

def run_test(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found in the current directory.")
        print(f"Current Directory: {os.getcwd()}")
        return

    print(f"--- Sending '{image_path}' to Jetson ({JETSON_IP}) ---")
    print("Note: The first request may take 15-30 seconds to warm up the GPU.")
    
    start_time = time.time()
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        try:
            # Added a 60-second timeout to prevent the laptop from giving up too early
            response = requests.post(URL, files=files, timeout=180)
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"\n[JETSON ANALYSIS] (Took {elapsed:.2f}s):")
                    print("-" * 30)
                    print(data['ai_response'])
                    print("-" * 30)
                else:
                    print(f"\n[SERVER ERROR]: {data.get('message', 'Unknown error')}")
            else:
                print(f"\n[HTTP ERROR]: Status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("\n[TIMEOUT]: The Jetson took too long to respond. Check if the model is stuck.")
        except requests.exceptions.ConnectionError:
            print("\n[CONNECTION ERROR]: Could not reach the Jetson. Check your WiFi/IP address.")
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR]: {e}")

if __name__ == "__main__":
    run_test(IMAGE_NAME)
