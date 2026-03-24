---
title: Indoor-MR-Assist-LV (FastVLM Unity)
emoji: 👓
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Real-time indoor navigation for low-vision users using FastVLM on Quest 3
pipeline_tag: visual-question-answering
library_name: unity-sentis
---

# Indoor-MR-Assist-LV: FastVLM 0.5B for Unity Sentis

This project is part of a research framework for on-device AI navigation for low-vision (LV) users on Meta Quest 3. It utilizes Unity 6 and Sentis (`com.unity.ai.inference`) for high-efficiency VLM inference.

## Project Phases
- **Phase 1 (On-Device):** Local inference using ExecuTorch/Sentis on Snapdragon XR2.
- **Phase 2 (Edge-Assisted):** Hybrid inference via a Flask-based bridge to a Jetson Orin Nano.

## Environment & Requirements

- **Unity Version**: `6000.3.6f1`
- **Sentis Version**: `com.unity.ai.inference 2.5.0` (customized)
  - **Custom layers added to the ONNX converter**: 
    `RotaryEmbedding`, `GroupQueryAttention`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`
  - **Implementation file**: 
    `FastVLM-Project/Packages/com.unity.ai.inference/Editor/ONNX/ONNXModelConverter.cs`

### Setup Note (Sentis Customization)
If the project does not run as-is, you must manually apply the custom layer support:
1. Download `com.unity.ai.inference` version `2.5.0`.
2. Copy the package from `Library/PackageCache/` to your local `Packages/` folder.
3. Overwrite `Editor/ONNX/ONNXModelConverter.cs` with the version provided in this repo.

## Project Structure

- `Assets/FastVLM/FastVLMScene.unity`: Main XR/Navigation scene.
- `Assets/FastVLM/VLMController.cs`: Bridge between camera/video input and inference loop.
- `Assets/FastVLM/ModelVLM.cs`: Model initialization and vision/text embedding composition.
- `Assets/FastVLM/Qwen2Tokenizer.cs`: Qwen2 BPE tokenizer.
- `Assets/StreamingAssets/fastvlm/`: Tokenizer config and vocabulary files.

## Required Model Files (ONNX)

Place the following ONNX files in `Assets/FastVLM/Models/` and assign them to the `ModelVLM` component:
Source: [FastVLM-0.5B-ONNX (HuggingFace)](https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/tree/main/onnx)

- `vision_encoder.onnx`
- `embed_tokens.onnx`
- `decoder_model_merged.onnx`

## Edge-Assisted Mode (Jetson Orin Nano)
For high-fidelity reasoning, this project can connect to an external inference server.
1. Navigate to `/FastVLM_Server`.
2. Run `python server.py`.
3. Update the `SERVER_URL` in Unity to match your Jetson's local IP (e.g., `10.0.0.172`).

## License
[MIT License](../LICENSE) | [Unity Companion License](https://unity3d.com/legal/licenses/unity_companion_license)

[Indoor-MR-Assist-LV] © 2026 Anthony Nguyen & Research Team
