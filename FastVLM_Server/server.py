import io
import os
import torch
import time
import threading
from flask import Flask, request, jsonify
from PIL import Image

# LLaVA Imports
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

app = Flask(__name__)

# --- CONFIG ---
MODEL_PATH = "./llava-fastvithd_0.5b_stage2"
DEVICE = "cuda"
CONV_MODE = "qwen_2"

model_lock = threading.Lock()
tokenizer, model, image_processor = None, None, None

def load_vlm():
    global tokenizer, model, image_processor
    print(f"--- Loading LLaVA to GPU (FP16 Mode) ---")
    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)
    
    # Load in Half-Precision (FP16) for Orin Nano Tensor Cores
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, 
        None, 
        model_name, 
        device=DEVICE,
        torch_dtype=torch.float16 
    )
    print("--- JETSON ORIN NANO READY ---")

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    with model_lock:
        start_time = time.time()
        try:
            if 'image' not in request.files:
                return jsonify({"status": "error", "message": "No image"})

            # Load and process image
            raw_image = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
            qs = "Describe what do you see in 2 sentences."

            # Move image to GPU and cast to Half-Precision
            image_tensor = process_images([raw_image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).to(DEVICE, dtype=torch.float16)

            # Build Prompt
            prompt = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs)
            conv = conv_templates[CONV_MODE].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            
            input_ids = tokenizer_image_token(
                conv.get_prompt(), 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(DEVICE)

            # GPU Inference with Explicit Stop Tokens
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, 
                    images=image_tensor, 
                    image_sizes=[raw_image.size], 
                    do_sample=False, 
                    max_new_tokens=64,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id, # Tells it where to stop
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode and strip special tokens
            ai_response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # --- THE "SPAM" KILLER ---
            # 1. Clean up known trailing tags
            for stop_word in ["<end of description>", "<|im_end|>", "###"]:
                ai_response = ai_response.split(stop_word)[0].strip()
            
            # 2. Prevent repetition loops
            ai_response = " ".join(dict.fromkeys(ai_response.split(". ")))
            
            elapsed = time.time() - start_time
            print(f"[LATENCY]: {elapsed:.2f}s | Response: {ai_response}")
            
            return jsonify({
                "status": "success", 
                "ai_response": ai_response,
                "latency": f"{elapsed:.2f}s"
            })

        except Exception as e:
            print(f"Server Error: {e}")
            return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    load_vlm()
<<<<<<< HEAD:FastVLM_Server/server_static_photo.py
    app.run(host='0.0.0.0', port=5000, debug=False)
=======
    app.run(host='0.0.0.0', port=5000, threaded=True)
>>>>>>> 354664f (feat: GPU-accelerated VLM server with TimeGhost (20MB) and Deco test suite):FastVLM_Server/server.py
