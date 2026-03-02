#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <vector>
#include <cstring>

using namespace ::executorch::extension;

extern "C" {
    static Module* global_module = nullptr;

    // Use the exact dimension from our Python export script
    const int INPUT_SIZE = 384; 
    const int CHANNELS = 3;

    bool LoadModel(const char* model_path) {
        if (global_module) {
            delete global_module;
        }
        // Initialize with the Vulkan-optimized .pte path
        global_module = new Module(model_path);
        auto err = global_module->load();
        return err == executorch::runtime::Error::Ok;
    }

    void RunInference(uint8_t* input_bytes, float* output_data) {
        if (!global_module) return;

        // FastVLM expects NCHW (1, 3, 384, 384)
        std::vector<float> float_input(1 * CHANNELS * INPUT_SIZE * INPUT_SIZE);

        // Professional Image Pre-processing (Normalization)
        // Unity provides RGBA; we need to convert to Planar RGB and Normalize
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; ++i) {
            // R-Channel
            float_input[0 * (INPUT_SIZE * INPUT_SIZE) + i] = (input_bytes[i * 4 + 0] / 255.0f - 0.485f) / 0.229f;
            // G-Channel
            float_input[1 * (INPUT_SIZE * INPUT_SIZE) + i] = (input_bytes[i * 4 + 1] / 255.0f - 0.456f) / 0.224f;
            // B-Channel
            float_input[2 * (INPUT_SIZE * INPUT_SIZE) + i] = (input_bytes[i * 4 + 2] / 255.0f - 0.406f) / 0.225f;
        }

        // Create the tensor wrapper
        auto tensor = from_blob(float_input.data(), {1, CHANNELS, INPUT_SIZE, INPUT_SIZE}, executorch::aten::ScalarType::Float);
        
        // Execute on Quest 3 GPU via Vulkan Delegate
        const auto result = global_module->forward(tensor);

        if (result.ok()) {
            auto out_tensor = result->at(0).toTensor();
            // out_tensor.nbytes() ensures we don't overflow the Unity buffer
            std::memcpy(output_data, out_tensor.const_data_ptr<float>(), out_tensor.nbytes());
        }
    }
}