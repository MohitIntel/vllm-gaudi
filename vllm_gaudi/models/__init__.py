from vllm import ModelRegistry

def register_model():
    from .gemma3 import GaudiGemma3ForCausalLM

    # For MODIFIED architectures: Use original name
    ModelRegistry.register_model(
        "Gemma3ForCausalLM",   # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3:GaudiGemma3ForCausalLM"
    )

    logger.info(f"###MD:model_arch: {model_arch} has been registered here!")