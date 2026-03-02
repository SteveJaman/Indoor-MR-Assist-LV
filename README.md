# Indoor-MR-Assist-LV
**On-device AI navigation for low-vision (LV) users on Meta Quest 3.**

This research framework implements a high-efficiency Vision Language Model (VLM) inference engine using **ExecuTorch** and **Vulkan**. The project focuses on Phase 1: benchmarking NPU/GPU latency, memory footprint, and thermal stability within a **Zero-Copy Unity-to-C++ pipeline** for real-time spatial awareness.

---

## Phase 1 Architecture: System Setup & Bridge
This phase establishes the foundational communication between the Unity XR environment and the underlying Snapdragon XR2 Gen 2 hardware.

### 1. Model Preparation (Step 1)
Converted and optimized the **FastVLM-0.5B** model for mobile edge deployment.
- **AOT Export:** Utilized `torch.export()` to capture the model graph.
- **Quantization & Lowering:** Applied `VulkanPartitioner` with `force_fp16` to leverage the Quest 3's GPU/NPU.
- **Serialization:** Generated optimized `.pte` (ExecuTorch) binaries for on-device loading.

### 2. Native C++ Bridge (Step 2)
Developed a high-performance C++ Native Plugin to bypass Unity's managed memory overhead.
- **Zero-Copy Tensor Mapping:** Direct memory access from Unity `Texture2D` to ExecuTorch Tensors.
- **Preprocessing:** Implemented Planar RGB normalization and resizing directly in C++ for sub-millisecond latency.
- **Hardware Abstraction:** Linked against the Qualcomm AI Engine via the ExecuTorch C++ API.

---

## Project Structure
* `Research_Pipeline/`: Python scripts for VLM export and Vulkan partitioning.
* `Native_Bridge/`: C++ source and CMake build system for the Android arm64-v8a bridge.
* `Unity_Integration/`: C# interface, XR HUD, and the model loader for the Quest 3 environment.

## Technical Achievements
| Metric | Status | Implementation |
| :--- | :--- | :--- |
| **Inference Engine** | ✅ Ready | ExecuTorch Runtime |
| **Memory Strategy** | ✅ Ready | Zero-Copy JNI Bridge |
| **Backend Target** | ✅ Ready | Vulkan / Qualcomm NPU |
| **Communication** | ✅ Ready | C# DllImport Heartbeat |

---

## Setup & Build
1. **Python:** `pip install torch executorch transformers`
2. **Native:** Compile `.cpp` via Android NDK (r27+) using the provided `CMakeLists.txt`.
3. **Unity:** Place compiled `.so` in `Assets/Plugins/Android/libs/arm64-v8a/`.
