# Indoor-MR-Assist-LV: Research Hub
**High-Efficiency AI Navigation for Low-Vision (LV) Users on Meta Quest 3**

This repository serves as a research framework for benchmarking Vision-Language Models (VLM) in indoor navigation. It contains multiple implementation paths, from pure on-device execution to hybrid edge-assisted architectures.

---

## 📂 Project Directory: Select Your Implementation

### 1. [FastVLM-Project (Unity Local Version)](./FastVLM-Project)
* **Engine:** Unity 6 + Sentis (`com.unity.ai.inference`)
* **Focus:** Real-time scene captioning using ONNX weights within the Unity environment.
* **Best for:** Rapid UI prototyping and testing Sentis custom layers.

### 2. [FastVLM_Server (Jetson Edge Bridge)](./FastVLM_Server)
* **Engine:** Python Flask + LLaVA/FastVLM 0.5B
* **Focus:** Offloading heavy inference to a **Jetson Orin Nano** via local network.
* **Best for:** High-fidelity spatial reasoning and testing "Video Segment" temporal logic.

### 3. [Native_Bridge & Research_Pipeline](./Native_Bridge)
* **Engine:** C++ Native Plugin + ExecuTorch + Vulkan
* **Focus:** The "Phase 1" Core—achieving zero-copy memory mapping and NPU acceleration.
* **Best for:** Production-level performance and hardware benchmarking.

---

## 🛠 Phase 1 Core Architecture: ExecuTorch + Vulkan
This baseline established the foundational communication between the Unity XR environment and the underlying Snapdragon XR2 Gen 2 hardware.

### 1. Model Preparation
Converted and optimized the **Apple FastVLM-0.5B** model for mobile edge deployment.
- **Vulkan Partitioning:** Applied `VulkanPartitioner` with `force_fp16` to leverage the Adreno GPU/NPU.
- **Serialization:** Generated optimized `.pte` binaries for the ExecuTorch runtime.

### 2. Native C++ Bridge
A high-performance C++ Native Plugin to bypass Unity's managed memory overhead.
- **Zero-Copy Tensor Mapping:** Direct memory access from Unity `Texture2D` to ExecuTorch Tensors via JNI.
- **Hardware Abstraction:** Linked against the Qualcomm AI Engine.

---

## 📊 Research Benchmarks & Status
| Implementation Path | Status | Target Backend | Latency Goal |
| :--- | :--- | :--- | :--- |
| **ExecuTorch Native** | ✅ Ready | Qualcomm NPU | < 500ms |
| **Unity Sentis** | 🏗 Testing | GPU (Vulkan) | < 1.5s |
| **Jetson Edge Bridge**| ✅ Ready | CPU/CUDA | < 3.0s (Network) |

---

## 🚀 Quick Start
Depending on which version you wish to test, navigate to the specific directory and follow the local `README.md`:

1.  **For Jetson Server:** `cd FastVLM_Server && python server.py`
2.  **For Unity Demo:** Open `FastVLM-Project` in Unity `6000.3.6f1`.
3.  **For Native Build:** Use the provided `CMakeLists.txt` in `Native_Bridge` with Android NDK r27+.

---

© 2026 Anthony Nguyen & Research Team | Licensed under MIT & Unity Companion License
