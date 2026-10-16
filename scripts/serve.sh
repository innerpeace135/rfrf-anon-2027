#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GPU_MEMORY="${GPU_MEMORY:-0.90}"
GPUS="${GPUS:-1}"
python -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name global_router --tensor-parallel-size "$GPUS" --port 8000 --host 127.0.0.1 --dtype bfloat16 --gpu-memory-utilization "$GPU_MEMORY" --guided-decoding-backend lm-format-enforcer
