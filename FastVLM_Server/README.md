# FastVLM Edge Inference Server
**High-Fidelity VLM Reasoning for Indoor Navigation via Jetson Orin Nano**

This server provides a REST API bridge for the `Indoor-MR-Assist-LV` Unity client. It offloads Vision-Language Model (VLM) inference to an NVIDIA Jetson edge device, allowing for deeper spatial reasoning than on-device mobile hardware.

---

## 🛠 Features
- **FastVLM-0.5B Backend:** Leverages the LLaVA-based Qwen2 architecture.
- **Manual KV Caching:** Optimized inference loop for CPU/CUDA stability.
- **Video Segment Support:** Accepts tiled frame grids for temporal scene analysis.
- **Flask REST API:** Simple POST interface for `Texture2D` data from Unity.

## 📦 Environment Setup

### 1. Requirements
- **Hardware:** NVIDIA Jetson Orin Nano (or any Linux/Windows PC with Python 3.10+).
- **Model Weights:** [Apple FastVLM-0.5B-Stage2](https://huggingface.co/apple/ml-fastvlm) (Place in `./llava-fastvithd_0.5b_stage2`).

### 2. Installation
```bash
# Clone LLaVA submodule if not present
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt
