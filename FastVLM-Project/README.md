---
title: fastvlm-0.5b-unity
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Real-time scene captioning with FastVLM ONNX on Unity Sentis
pipeline_tag: visual-question-answering
library_name: unity-sentis
---

# FastVLM 0.5B for Unity Sentis

This repository is a Unity 6 + Sentis (`com.unity.ai.inference`) demo for FastVLM-based scene captioning.

## Demo (YouTube)

[![FastVLM Unity](https://img.youtube.com/vi/zyDNLEEXR0Q/0.jpg)](https://www.youtube.com/watch?v=zyDNLEEXR0Q)

## Environment

- **Unity Version**: `6000.3.6f1`
- **Sentis Version**: `com.unity.ai.inference 2.5.0` (customized)
  - **Custom layers added to the ONNX converter**:
    `RotaryEmbedding`, `GroupQueryAttention`, `SimplifiedLayerNormalization`, `SkipSimplifiedLayerNormalization`
  - **Implementation file**:
    `fastvlm-0.5b-unity/Packages/com.unity.ai.inference/Editor/ONNX/ONNXModelConverter.cs`

### Setup Note (If It Does Not Run As-Is)

If the project does not run with the current setup:

1. Download `com.unity.ai.inference` version `2.5.0`.
2. Copy `Library/PackageCache/com.unity.ai.inference@xxxxxx` to `Packages/com.unity.ai.inference`.
3. Overwrite only this file:
   `com.unity.ai.inference/Editor/ONNX/ONNXModelConverter.cs`

## Project Structure

- `Assets/FastVLM/FastVLMScene.unity`: Main runtime scene
- `Assets/FastVLM/VLMController.cs`: VideoPlayer-UI bridge and continuous inference loop
- `Assets/FastVLM/ModelVLM.cs`: Model initialization, vision/text embedding composition, and generation
- `Assets/FastVLM/Qwen2Tokenizer.cs`: Qwen2 BPE tokenizer
- `Assets/StreamingAssets/fastvlm/`: `vocab.json`, `merges.txt`, `tokenizer_config.json`

## Required Model Files

Prepare the ONNX files below in `Assets/FastVLM/Models/` and assign them to the `ModelVLM` component in `VLMManager`.

Source models:  
https://huggingface.co/onnx-community/FastVLM-0.5B-ONNX/tree/main/onnx

Download the three files below from the link above, then copy them into `Assets/FastVLM/Models/`.

- `vision_encoder.onnx`
- `embed_tokens.onnx`
- `decoder_model_merged.onnx`

## Quick Start

1. Open the project in Unity `6000.3.6f1`.
2. Open `Assets/FastVLM/FastVLMScene.unity`.
3. Check `VLMManager > ModelVLM` and verify all `ModelAsset` fields are assigned.
4. Hit Play.
5. Edit the prompt in `InputField` if needed. The next loop uses the updated prompt.

## License

[Unity Companion License](https://unity3d.com/legal/licenses/unity_companion_license)

[fastvlm-0.5b-unity] © 2026 Unity Technologies

Licensed under the Unity Companion License for Unity-dependent projects (see https://unity3d.com/legal/licenses/unity_companion_license).
Unless expressly provided otherwise, the Software under this license is made available strictly on an "AS IS" BASIS WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. Please review the license for details on these and other terms and conditions.