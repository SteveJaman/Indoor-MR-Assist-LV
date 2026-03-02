import torch
import os
import logging
import shutil
from transformers import AutoModelForCausalLM
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def sync_schemas():
    """CMD Fix: Ensures .fbs files are in the _serialize directory."""
    target_dir = os.path.join("executorch", "exir", "_serialize")
    source_dir = os.path.join("executorch", "schema")
    
    if os.path.exists(source_dir):
        logger.info("Synchronizing environment schema files...")
        for file in os.listdir(source_dir):
            if file.endswith(".fbs"):
                shutil.copy(os.path.join(source_dir, file), target_dir)
        return True
    return False

def main():
    # Run environment sync before starting
    sync_schemas()

    model_id = "apple/FastVLM-0.5B"
    output_path = "fastvlm_vision_vulkan.pte"
    
    # 1. Load Model
    logger.info(f"Loading weights for {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    # 2. Extract & Prep Vision Tower
    # Cast to float32 to ensure compatibility with Adreno GPU [cite: 3, 12, 13]
    logger.info("Preparing Vision Tower (float32)...")
    vision_tower = model.get_vision_tower().to(torch.float32)
    example_inputs = (torch.randn(1, 3, 384, 384).to(torch.float32),)

    # 3. Export to Graph
    logger.info("Capturing program graph...")
    exported_program = torch.export.export(vision_tower, example_inputs, strict=False)

    # 4. Vulkan Partitioning
    # force_fp16 is essential for Quest 3 hardware performance 
    logger.info("Partitioning for Vulkan Backend...")
    vulkan_partitioner = VulkanPartitioner(compile_options={"force_fp16": True})

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[vulkan_partitioner]
    )

    # 5. Final Serialization
    logger.info(f"Serializing to {output_path}...")
    et_program = edge_program.to_executorch()
    
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    logger.info("==========================================")
    logger.info("SUCCESS: .pte file is ready for Quest 3.")
    logger.info("==========================================")

if __name__ == "__main__":
    main()