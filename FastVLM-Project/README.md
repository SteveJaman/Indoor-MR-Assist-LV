---
title: Indoor-MR-Assist-LV (FastVLM Unity Local)
emoji: 👓
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Local FastVLM scene captioning on Unity Sentis for Meta Quest 3
pipeline_tag: visual-question-answering
library_name: unity-sentis
---

# Indoor-MR-Assist-LV: FastVLM 0.5B for Unity Sentis

This directory contains the **Local On-Device** implementation for the Indoor-MR-Assist research framework. This version utilizes Unity 6 and a customized Sentis (`com.unity.ai.inference`) runtime to execute FastVLM 0.5B directly on the Meta Quest 3 (Snapdragon XR2 Gen 2).

## Project Role
This folder represents **Phase 1** of our research: benchmarking local inference latency and on-device spatial awareness. For high-fidelity off-device reasoning, see the [Jetson Edge Server](../FastVLM_Server).

---

## Environment & Requirements

- **Unity Version**: `6000.3.6f1`
- **Sentis Version**: `com.unity.ai.inference 2.5.0` (customized)
- **Customization**: This project requires specific ONNX converter layers:
  - `RotaryEmbedding`, `GroupQueryAttention`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`.
  - **Implementation file**: `FastVLM-Project/Packages/com.unity.ai.inference/Editor/ONNX/ONNXModelConverter.cs`

### Setup Note (Sentis Customization)
If the project fails to import the ONNX models:
1. Download `com.unity.ai.inference` version `2.5.0`.
2. Move the package from `Library/PackageCache/` to your project's local `Packages/` folder.
3. Overwrite `Editor/ONNX/ONNXModelConverter.cs` with the version provided in this repository.

---

## Project Structure

- `Assets/FastVLM/FastVLMScene.unity`: Main XR runtime scene.
- `Assets/FastVLM/VLMController.cs`: Handles video/camera input and the continuous inference loop.
- `Assets/FastVLM/ModelVLM.cs`: Manages model initialization and embedding composition.
- `Assets/FastVLM/Qwen2Tokenizer.cs`: BPE tokenizer for Qwen2.
- `Assets/StreamingAssets/fastvlm/`: Vocabulary and tokenizer configuration files.

## Required Model Files (ONNX)

Download and place the following files in `Assets/FastVLM/Models/`:
Source: [FastVLM-0.5B-ONNX (HuggingFace)](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/tree/main/onnx)

- `vision_encoder.onnx`
- `embed_tokens.onnx`
- `decoder_model_merged.onnx`

---

## Credits & Attribution
This implementation is a replication and research adaptation of the work by **Sky-Kim**.
- **Original Author**: Sky-Kim ([sky-kim-fastvlm-unity](https://github.com/Sky-Kim))
- **Original License**: Unity Companion License

## License
[MIT License](../LICENSE) | [Unity Companion License](https://unity3d.com/legal/licenses/unity_companion_license)

[Indoor-MR-Assist-LV] © 2026 Anthony Nguyen & Research Team
