# Club2080Ti

Recipes, patches, benchmarks, and failure notes for running modern LLM
inference stacks on RTX 2080 Ti / Turing SM75.

This repository is a working toolbox, not an official support matrix. The first
target is dual RTX 2080 Ti with NVLink on `miniclaw`, but the notes should also
help other SM75 users understand which modern inference paths are realistic.

## Current Best Route

`Qwen3.6-27B-AWQ` on dual RTX 2080 Ti:

- Engine: vLLM 0.21
- GPUs: 2x RTX 2080 Ti, tensor parallel size 2
- Quantization: AWQ Marlin
- Attention: FlashInfer full attention
- GDN prefill: FlashQLA legacy backend
- Speculative decode: Qwen3.5/3.6 MTP, best current setting `K=3`
- Status: usable in benchmark runs; best current Club2080Ti route

Measured highlights:

| Workload | Prefill | Decode | E2E | Notes |
| --- | ---: | ---: | ---: | --- |
| 4K / tg128, MTP K=3 | `1843.7 tok/s` | `79.14 tok/s` | `3.839s` | Peak 4K sweep result |
| 64K / tg512 cap, MTP K=3 | `1294.3 tok/s` | `55.33 tok/s` | `56.768s` | Completion ended at 405 tokens; not strict ignore-eos |
| Ragent6 0.2.2 zh-CN | - | - | `167.39s wall` | strict `43/60`, partial weighted `82.5/100` |

See [models/qwen3.6-27b-awq/vllm-mtp-k3.md](models/qwen3.6-27b-awq/vllm-mtp-k3.md).

## Experimental Route

`Qwen3.6-27B-AWQ` on SGLang reached a short `/generate` HTTP 200 on dual
2080 Ti using SGLang + FlashInfer + FlashQLA legacy + SM75 fallbacks.

This is not production-ready:

- Short smoke returned `prompt_tokens=5`, `completion_tokens=2`, `e2e_latency=3.67s`.
- Output was `ieee!`, not the expected `OK`.
- No reliable throughput or quality benchmark has been completed.

See [models/qwen3.6-27b-awq/sglang-smoke.md](models/qwen3.6-27b-awq/sglang-smoke.md).

## Repository Map

- [engines/vllm](engines/vllm/README.md): vLLM SM75 recipes and launch notes.
- [engines/sglang](engines/sglang/README.md): SGLang experimental SM75 patch log.
- [engines/llamacpp](engines/llamacpp/README.md): llama.cpp baselines.
- [engines/flashqla](engines/flashqla/README.md): FlashQLA SM70/SM75 kernel notes.
- [models](models/README.md): model-specific compatibility notes.
- [BENCHMARKS.md](BENCHMARKS.md): curated scorecard.
- [STATUS.md](STATUS.md): working, experimental, and rejected paths.
- [scripts](scripts/): small utilities for snapshots and repeatable benchmark calls.
- [manifests](manifests/): environment and workload locks.
- [reports](reports/): curated summaries and raw log pointers.

## Non-Goals For The First Version

- No model weights in git.
- No vendored copies of vLLM, SGLang, FlashInfer, or FlashQLA.
- No claim of official upstream support.
- No one-command installer that hides the actual patch set.

## Source Notes

Initial numbers come from local miniclaw experiments recorded on 2026-05-17 to
2026-05-19 under `/home/max/memory`, `/home/max/reports`, and
`/home/max/Develop/Ragent6/reports`.

