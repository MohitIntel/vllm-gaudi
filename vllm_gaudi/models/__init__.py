from vllm import ModelRegistry

def register_model():
    from .gemma3 import GaudiGemma3ForCausalLM
    from .gemma3_mm import GaudiGemma3ForConditionalGeneration

    # For MODIFIED architectures: Use original name
    ModelRegistry.register_model(
        "Gemma3ForCausalLM",   # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3:GaudiGemma3ForCausalLM"
    )

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",   # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:GaudiGemma3ForConditionalGeneration"
    )

    logger.info(f"###MD:model_arch: {model_arch} has been registered here!")