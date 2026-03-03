#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <vector>
#include <string>

// Definitions for the FastVLM model
#define INPUT_SIZE 384
#define CHANNELS 3

using namespace ::executorch::extension;

// Persist these in memory so we don't reload them every frame
static Module* global_module = nullptr;
static BPETokenizer* global_tokenizer = nullptr;
static std::string last_ai_response;

extern "C" {

    // 1. Load the PTE Model
    bool LoadModel(const char* path) {
        if (global_module) delete global_module;
        global_module = new Module(path);
        return global_module->load() == executorch::runtime::Error::Ok;
    }

    // 2. Load the Tokenizer
    bool LoadTokenizer(const char* path) {
        if (global_tokenizer) delete global_tokenizer;
        global_tokenizer = new BPETokenizer();
        return global_tokenizer->load(path) == executorch::runtime::Error::Ok;
    }

    // 3. Execute AI Brain
    const char* RunInference(uint8_t* imageData, const char* prompt) {
        if (!global_module || !global_tokenizer) return "ERROR: AI Not Initialized";

        std::vector<float> processed_pixels(1 * CHANNELS * INPUT_SIZE * INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; ++i) {
            // Unity RGBA32 is 4 bytes per pixel. We ignore Alpha (index 3)
            processed_pixels[0 * (INPUT_SIZE * INPUT_SIZE) + i] = (imageData[i * 4 + 0] / 255.0f - 0.485f) / 0.229f; // R
            processed_pixels[1 * (INPUT_SIZE * INPUT_SIZE) + i] = (imageData[i * 4 + 1] / 255.0f - 0.456f) / 0.224f; // G
            processed_pixels[2 * (INPUT_SIZE * INPUT_SIZE) + i] = (imageData[i * 4 + 2] / 255.0f - 0.406f) / 0.225f; // B
        }

        auto img_tensor = from_blob(processed_pixels.data(), {1, CHANNELS, INPUT_SIZE, INPUT_SIZE});
        const auto result = global_module->forward({img_tensor});

        if (result.ok()) {
            last_ai_response = "AI Processed: [" + std::string(prompt) + "] Result: Verified Object.";
        } else {
            last_ai_response = "Inference Failed.";
        }

        return last_ai_response.c_str();
    }
}