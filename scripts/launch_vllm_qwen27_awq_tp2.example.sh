#!/usr/bin/env bash
set -euo pipefail

# Example only. Replace MODEL with a local path or HF id.
MODEL="${MODEL:-/path/to/Qwen3.6-27B-AWQ}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.86 \
  --quantization awq_marlin \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
