# 2080Ti LLM Toolbox

Recipes, patches, benchmarks, and failure notes for running modern LLM
inference stacks on RTX 2080 Ti / Turing SM75.

This repository is a working toolbox, not an official support matrix. The first
target is a dual RTX 2080 Ti 22GB NVLink system, but the notes should also help
other SM75 users understand which modern inference paths are realistic.

## Tested Hardware

The headline results are not from a stock 11GB card. The current test rig uses:

- 2x modified RTX 2080 Ti 22GB cards
- NVLink enabled between the two cards
- one TU102-300A GPU and one TU102-300 GPU
- single-request serving tests unless a row explicitly says otherwise

Single-request means one active request at a time. This repository keeps that
separate from multi-user batching throughput so vLLM, SGLang, and llama.cpp can
be compared on the same practical serving shape.

## Current Best Route

`Qwen3.6-27B-AWQ` on dual RTX 2080 Ti:

- Engine: vLLM 0.21
- GPUs: 2x RTX 2080 Ti 22GB over NVLink, tensor parallel size 2
- Quantization: AWQ Marlin
- Attention: FlashInfer full attention
- GDN prefill: FlashQLA SM70/SM75 legacy backend from
  [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)
- Speculative decode: Qwen3.5/3.6 MTP, best current setting `K=3`
- Status: usable in single-request benchmark runs; best current route

Measured highlights:

| Workload | Prefill | Decode | E2E | Notes |
| --- | ---: | ---: | ---: | --- |
| 4K / tg128, MTP K=3 | `1843.7 tok/s` | `79.14 tok/s` | `3.839s` | Peak 4K sweep result |
| 64K / tg512 cap, MTP K=3 | `1294.3 tok/s` | `55.33 tok/s` | `56.768s` | Completion ended at 405 tokens; not strict ignore-eos |
| 60-request serving run | `700.9 tok/s avg` | `35.2 tok/s avg` | `167.39s wall` | Sequential real requests, not concurrent batching |

See [models/qwen3.6-27b-awq/vllm-mtp-k3.md](models/qwen3.6-27b-awq/vllm-mtp-k3.md).

## Experimental Route

`Qwen3.6-27B-AWQ` on SGLang reached a short `/generate` HTTP 200 on dual
2080 Ti using SGLang + FlashInfer + our FlashQLA SM70/SM75 legacy backend +
SM75 fallbacks.

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

Initial numbers come from private lab notes and reports from 2026-05-17 to
2026-05-19. Public-safe summaries are copied into this repository; machine-local
paths are intentionally omitted.
