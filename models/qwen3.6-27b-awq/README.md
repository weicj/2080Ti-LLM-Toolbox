# Qwen3.6-27B-AWQ

This was the first main target for 2080Ti LLM Toolbox. The original resident
AWQ weights are now retired from the lab rig: Qwopus AWQ is the current Qwen-family
quality lane, and Heretic GPTQ is the low-refusal lane.

Historical benchmark rows in this directory are retained for debugging and
regression comparison. They have not been rerun on the final 2026-05-30 repaired
stable build that fixed `TQ4NC + MTP3 + no-eager/CUDAGraph`.

## Recommended

- [vLLM best SM75 route](../../engines/vllm/recipes/qwen36-27b-awq-best-sm75.md)
- [vLLM MTP K=3](vllm-mtp-k3.md)

## Resident Model Status

- Former resident checkpoint: `QuantTrio-Qwen3.6-27B-AWQ`
- Status: removed from the resident lab model set on 2026-05-30 to recover disk
  space
- Replacement quality lane: `mconcat/Qwopus3.6-27B-v2-AWQ-4bit`
- Replacement low-refusal lane:
  `Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`
- Re-download and file-manifest record: private lab registry note dated
  2026-05-24

## Experimental

- [vLLM TurboQuant KV](vllm-turboquant-kv.md)
- [SGLang smoke](sglang-smoke.md)

## Baseline

- [llama.cpp baseline](llamacpp-baseline.md)
