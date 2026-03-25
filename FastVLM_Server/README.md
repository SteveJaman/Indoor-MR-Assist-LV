# FastVLM Edge Inference Server
**High-Fidelity VLM Reasoning for Indoor Navigation via Jetson Orin Nano**

This server provides a REST API bridge for the `Indoor-MR-Assist-LV` Unity client. It offloads Vision-Language Model (VLM) inference to an NVIDIA Jetson edge device, allowing for deeper spatial reasoning than on-device mobile hardware.

---

## 🛠 Features
- **FastVLM-0.5B Backend:** Leverages the LLaVA-based Qwen2 architecture.
- **GPU Acceleration:** Optimized inference latency reduced from ~42s to **~6.5s** on Jetson hardware.
- **Manual KV Caching:** Optimized inference loop for CPU/CUDA stability.
- **Video Segment Support:** Accepts tiled frame grids for temporal scene analysis (e.g., Hazard Detection).
- **Flask REST API:** Simple POST interface for `Texture2D` data from Unity.

## 📦 Environment Setup

### 1. Requirements
- **Hardware:** NVIDIA Jetson Orin Nano (or any Linux/Windows PC with Python 3.10+).
- **External Assets (Not in Git):**
    - **Model Weights:** [Apple FastVLM-0.5B-Stage2](https://huggingface.co/apple/ml-fastvlm) (Place in `./llava-fastvithd_0.5b_stage2`).
    - **Critical File:** Place `vision_encoder.onnx` (505MB) in the root directory.
    - **OpenCV:** Ensure OpenCV 4.x is installed on the host system.
- **Test Samples:** Place test videos (e.g., `TimeGhost.mp4`) in the `samples/` folder for evaluation.

### 2. Installation
```bash
# 1. Clone the repository
git clone [https://github.com/SteveJaman/Indoor-MR-Assist-LV.git](https://github.com/SteveJaman/Indoor-MR-Assist-LV.git)
cd FastVLM_Server

# 2. Install dependencies (Optimized for Jetson)
pip install -r requirements_jetson.txt

# 3. Add External Models
# Ensure 'vision_encoder.onnx' and 'llava-fastvithd_0.5b_stage2/' 
# are present in the root folder.
```
### To initialize the VLM and start the inference bridge:
```PowerShell
python server.py
```

### To run the evaluation client against the server:
```PowerShell
python vlm_client_eval.py
```
