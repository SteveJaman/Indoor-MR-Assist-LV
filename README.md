# Indoor-MR-Assist-LV: Research Hub
**AI-Powered Indoor Navigation for Low-Vision (LV) Users on Meta Quest 3**

This repository is a research framework investigating Vision-Language Model (VLM) deployment strategies for spatial awareness. It currently supports two primary inference paths: Local On-Device (Unity Sentis) and Edge-Assisted (Jetson Orin Nano).

---

## 📂 Project Directory

### 1. [FastVLM-Project (Unity Local Version)](./FastVLM-Project)
* **Engine:** Unity 6 + Sentis (`com.unity.ai.inference`)
* **Focus:** On-device scene captioning using optimized ONNX weights.
* **Status:** ✅ Functional Demo with `FastVLMScene.unity`.
* **Hardware:** Runs directly on Meta Quest 3 / Windows PC.

### 2. [FastVLM_Server (Jetson Edge Bridge)](./FastVLM_Server)
* **Engine:** Python Flask + LLaVA/FastVLM 0.5B
* **Focus:** High-fidelity off-device reasoning and temporal video analysis.
* **Status:** ✅ Functional API with `test_jetson.py` validation.
* **Hardware:** NVIDIA Jetson Orin Nano (Edge Server).

---

## 🧪 Research Objectives
The goal of this project is to compare the trade-offs between **Latency** (On-device) and **Reasoning Depth** (Edge-assisted) for assisting users with visual impairments.

### Technical Achievements (Phase 1)
| Feature | Implementation | Status |
| :--- | :--- | :--- |
| **VLM Architecture** | Apple FastVLM-0.5B | ✅ Integrated |
| **Unity Inference** | Sentis ONNX Runtime | ✅ Operational |
| **Network Bridge** | Flask REST API | ✅ Operational |
| **Video Sampling** | Tiled Keyframe Processing | 🏗️ In Testing |

---

## 🚀 Quick Start

1.  **Clone the Repo:** `git clone https://github.com/SteveJaman/Indoor-MR-Assist-LV`
2.  **To Run Jetson Server:** * `cd FastVLM_Server`
    * `python server.py`
3.  **To Run Unity Project:** * Open the `FastVLM-Project` folder in Unity `6000.3.6f1`.
    * Ensure ONNX models are in `StreamingAssets`.

---

© 2026 Anthony Nguyen | Kennesaw State University MSSE Research
