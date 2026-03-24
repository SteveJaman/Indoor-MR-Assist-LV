import io
import os
import sys
import torch
from flask import Flask, request, jsonify
from PIL import Image
from tqdm import tqdm

# LLaVA Imports
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

app = Flask(__name__)

# --- CONFIG ---
MODEL_PATH = "./llava-fastvithd_0.5b_stage2"
DEVICE = "cpu"
CONV_MODE = "qwen_2"

# Global Model Variables
tokenizer = None
model = None
image_processor = None
model_name = None

def load_vlm():
    global tokenizer, model, image_processor, model_name
    
    # 1. Generation Config Safety Toggle
    # Research models sometimes have conflicting configs; this ensures a clean load
    gen_config_path = os.path.join(MODEL_PATH, 'generation_config.json')
    temp_config_path = os.path.join(MODEL_PATH, '.generation_config.json')
    if os.path.exists(gen_config_path):
        os.rename(gen_config_path, temp_config_path)

    try:
        print(f"--- Loading model via Builder from {MODEL_PATH} ---")
        disable_torch_init()
        model_name = get_model_name_from_path(MODEL_PATH)
        
        # This function handles the LlavaQwen2 / MobileCLIP mapping automatically
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            MODEL_PATH, None, model_name, device=DEVICE
        )
        
        # Stability for Jetson CPU
        model = model.float()
        print("--- SUCCESS: JETSON SERVER IS LIVE ---")

    finally:
        # Restore config so the directory remains intact
        if os.path.exists(temp_config_path):
            os.rename(temp_config_path, gen_config_path)

@app.route('/detect_hazards', methods=['POST'])
def detect_hazards():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image"})

        # 1. Process Image
        raw_image = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        
        # Optimization: Resize to save CPU cycles (from your reference code)
        if max(raw_image.size) > 512:
            raw_image.thumbnail((512, 512))
            
        image_tensor = process_images([raw_image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).float().to(DEVICE)

        # 2. Build Prompt
        qs = "Describe the characters and colors in this image."
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)

        # 3. Manual Inference Loop with KV Caching & Progress Bar
        max_new_tokens = 64 # Shorter for testing, can be 256
        output_ids = input_ids.clone()
        past_key_values = None
        
        print(f"[INFERENCE] Thinking...")
        pbar = tqdm(total=max_new_tokens, desc="Generating", unit="token", leave=False)

        with torch.inference_mode():
            for i in range(max_new_tokens):
                if past_key_values is None:
                    # Initial pass
                    inputs = {
                        "input_ids": output_ids,
                        "images": image_tensor,
                        "image_sizes": [raw_image.size],
                        "use_cache": True
                    }
                else:
                    # Efficient caching pass
                    inputs = {
                        "input_ids": output_ids[:, -1:],
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }

                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                # Greedy Decoding (Mirroring your logic)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                output_ids = torch.cat([output_ids, next_token], dim=-1)
                past_key_values = outputs.past_key_values
                pbar.update(1)

                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        pbar.close()
        ai_response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[RESULT]: {ai_response}")

        return jsonify({"status": "success", "ai_response": ai_response})

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    load_vlm()
    app.run(host='0.0.0.0', port=5000, debug=False)