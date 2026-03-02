# Indoor-MR-Assist-LV
**On-device AI navigation for low-vision (LV) users on Meta Quest 3.**

This research framework implements a high-efficiency Vision Language Model (VLM) inference engine using **ExecuTorch** and **Vulkan**. The project focuses on Phase 1: benchmarking NPU/GPU latency, memory footprint, and thermal stability within a **Zero-Copy Unity-to-C++ pipeline** for real-time spatial awareness.

**Target Hardware:** Qualcomm Snapdragon XR2 Gen 2 (Meta Quest 3)

---

## Phase 1 Architecture: System Setup & Bridge
This phase establishes the foundational communication between the Unity XR environment and the underlying hardware.

### 1. Model Preparation (Step 1)
Converted and optimized the **Apple FastVLM-0.5B** model for mobile edge deployment.
- **AOT Export:** Utilized `torch.export()` and `exir` to capture the model graph.
- **Vulkan Partitioning:** Applied `VulkanPartitioner` with `force_fp16` to leverage the Adreno GPU/NPU.
- **Serialization:** Generated optimized `.pte` binaries with custom schema synchronization for ExecuTorch runtime.

### 2. Native C++ Bridge (Step 2)
Developed a high-performance C++ Native Plugin to bypass Unity's managed memory overhead.
- **Zero-Copy Tensor Mapping:** Direct memory access from Unity `Texture2D` to ExecuTorch Tensors via JNI.
- **Preprocessing:** Implemented Planar RGB normalization $(x/255.0 - 0.485)/0.229$ directly in C++.
- **Hardware Abstraction:** Linked against the Qualcomm AI Engine via the ExecuTorch C++ Runtime API.

---

## Project Structure
* `Research_Pipeline/`: Python scripts for VLM export, `requirements.txt`, and Vulkan partitioning logic.
* `Native_Bridge/`: C++ source, `CMakeLists.txt` build system, and Android NDK configurations.
* `Unity_Integration/`: C# interface logic, XR HUD, and the Model Loader for Quest 3.

## Technical Achievements
| Metric | Status | Implementation |
| :--- | :--- | :--- |
| **Inference Engine** | ✅ Ready | ExecuTorch Runtime |
| **Memory Strategy** | ✅ Ready | Zero-Copy JNI Bridge |
| **Backend Target** | ✅ Ready | Vulkan / Qualcomm NPU |
| **Communication** | ✅ Ready | C# DllImport Heartbeat |

---

## Setup & Build
1. **Python:** `pip install -r requirements.txt` and run `python export_fastvlm.py`.
2. **Native:** Compile `.cpp` via Android NDK (r27+) using the provided `CMakeLists.txt`.
3. **Unity:** Place compiled `.so` binaries in `Assets/Plugins/Android/libs/arm64-v8a/` and the `.pte` model in `StreamingAssets`.
