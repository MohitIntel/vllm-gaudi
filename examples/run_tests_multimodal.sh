# !/bin/bash

VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 pytest -v -s ../tests/models/multimodal/generation/test_common.py --model_card_path ../tests/models/multimodal/generation/model_cards/Llama-4-Scout-17B-16E-Instruct-vision.yaml

##VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Qwen3-30B-A3B.yaml