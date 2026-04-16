import sys, os, gc, io, time, threading, cv2
import numpy as np
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from tqdm import tqdm
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
VLM_PATH = str(BASE_DIR / "llava-fastvithd_1.5b_stage3")

HOST_IP = '0.0.0.0' 
PORT = 5000

model_lock = threading.Lock() 
DEVICE = "cpu" 
TORCH_DTYPE = torch.bfloat16 

# --- OFFICIAL STREAMER ---
class VisualStreamer(TextStreamer):
    def __init__(self, tokenizer, total_tokens):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.pbar = tqdm(total=total_tokens, desc="[*] Generating", unit="tk", leave=False)
    
    def put(self, value):
        if value.size(0) == 1:
            self.pbar.update(1)
        super().put(value)

    def end(self):
        self.pbar.close()
        super().end()

# --- MODEL LOADING ---
def load_engines():
    global tokenizer, vlm_model, image_processor
    disable_torch_init()
    
    print(f"[*] Loading LLaVA 1.5B (Official Qwen-2 Logic)...")
    model_name = get_model_name_from_path(VLM_PATH)
    
    # Official loader handles low_cpu_mem_usage internally
    tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        VLM_PATH, 
        None, 
        model_name, 
        device=DEVICE, 
        torch_dtype=TORCH_DTYPE 
    )
    
    vlm_model.eval()
    print(f"\n✓ VLM SYSTEM READY ON {DEVICE.upper()}\n")

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    if model_lock.locked():
        return jsonify({"status": "busy"}), 429

    with model_lock:
        start_time = time.time()
        img_file = request.files.get('image')
        if not img_file: return jsonify({"status": "error"}), 400

        try:
            # 1. Image Processing
            img_bytes = img_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            
            # Surface Pro Optimization: 336 is standard, 224 is 'Fast'
            vlm_img = pil_img.resize((224, 224)) 
            image_tensor = process_images([vlm_img], image_processor, vlm_model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE, dtype=TORCH_DTYPE)

            # 2. Proper Conversation Formatting (Official Way)
            conv = conv_templates["qwen_2"].copy()
            roles = conv.roles
            
            # Format: <|im_start|>user\n<image>\nDescribe path.<|im_end|>\n<|im_start|>assistant\n
            prompt_message = DEFAULT_IMAGE_TOKEN + "\nDescribe the path and any obstacles you see."
            conv.append_message(roles[0], prompt_message)
            conv.append_message(roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)

            # 3. Execution
            max_new_tokens = 20
            streamer = VisualStreamer(tokenizer, max_new_tokens)

            print("[*] Processing Prefill (Image + Prompt)...")
            with torch.inference_mode():
                output_ids = vlm_model.generate(
                    input_ids, 
                    images=image_tensor, 
                    image_sizes=[vlm_img.size],
                    do_sample=False, # Greedy decoding is fastest for CPU
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=streamer
                )
            
            # Decode only the NEW tokens
            vlm_context = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            latency = int((time.time() - start_time) * 1000)
            print(f"\n[SUCCESS] {latency}ms | {vlm_context}")

            return jsonify({
                "status": "success", 
                "final_response": vlm_context, 
                "latency": latency
            })

        except Exception as e:
            print(f"\nVLM ERROR: {e}")
            return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    load_engines()
    app.run(host=HOST_IP, port=PORT, debug=False, threaded=True)