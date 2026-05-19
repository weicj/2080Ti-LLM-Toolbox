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

## Framework Status

This toolbox is centered on three inference frameworks on RTX 2080 Ti / SM75:
vLLM, llama.cpp, and SGLang. They are deliberately compared with
single-request serving numbers, not blended with multi-user batching results.

| Framework | Current Status | Main Tested Model | Best Use Right Now |
| --- | --- | --- | --- |
| vLLM | Recommended route | `Qwen3.6-27B-AWQ` | Fast single-request serving on dual 22GB 2080 Ti with TP=2, FlashInfer, FlashQLA, AWQ Marlin, and MTP K=3 |
| llama.cpp | Reliable baseline | Qwen3.6/Qwopus 27B GGUF | Practical single-card baseline and fallback; slower long-prefill than vLLM, but simple and robust |
| SGLang | Experimental | `Qwen3.6-27B-AWQ` | Compatibility research and patch archive; not ready for production comparison |

FlashQLA is the kernel foundation for the vLLM and SGLang GDN path here. The
SM70/SM75 backend is maintained separately at
[weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75).

## Mature Framework Comparison

The table below only compares routes with usable performance data. SGLang is
tracked separately because the current work is still compatibility bring-up, not
a production-serving result.

Same-class 27B Qwen3.6-family single-request serving measurements:

| Framework | Config | Prompt / Generate | Prefill | Decode | E2E | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vLLM | Qwen3.6-27B-AWQ, TP=2, MTP K=3 | 4K / 128 | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` | Best current route |
| vLLM | Qwen3.6-27B-AWQ, TP=2, MTP K=3 | 64K / cap | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` | Best current long-context route |
| llama.cpp | 27B GGUF, single 2080 Ti | 4114 / 128 | `553.4 tok/s` | `23.7 tok/s` | `12.8s` | Baseline |
| llama.cpp | 27B GGUF, single 2080 Ti | 64022 / 512 | `383.1 tok/s` | `16.3 tok/s` | `198.6s` | Baseline |

Interpretation:

- vLLM is the current lead for the dual-card 22GB setup, especially long-context
  prefill and MTP-assisted decode.
- llama.cpp is still the practical baseline and the sanity check for every
  optimization claim.

## Sequential 60-Request Serving Run

For agent-style workload testing, this repository uses a 60-request sequential
run: real requests are sent one after another to the same server. This measures
single-request serving behavior under repeated use without hiding latency inside
concurrent batching.

| Framework | Config | Wall Time | Avg Prefill | Avg Decode | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| vLLM | Qwen3.6-27B-AWQ, TP=2, MTP K=3 | `167.4s` | `700.9 tok/s` | `35.2 tok/s` | Current best validated route |
| llama.cpp | 27B GGUF baseline | `471.0s` | `350.3 tok/s` | `21.2 tok/s` | Earlier same-style run |
| llama.cpp | 27B GGUF MTP n=2 | `306.0s` | `297.0 tok/s` | `45.1 tok/s` | Faster decode, prefill penalty |

Model scores from these runs are treated as sanity checks only. This repository
is about whether the 2080 Ti serving stack runs fast and correctly; benchmark
scores mostly reflect the model.

## SGLang SM75 Bring-Up

SGLang is tracked as a separate experimental effort. It should not be read as a
peer of the vLLM and llama.cpp rows above yet.

What works now:

- `Qwen3.6-27B-AWQ` reached a short `/generate` HTTP 200 on dual RTX 2080 Ti.
- The smoke returned `prompt_tokens=5`, `completion_tokens=2`, and
  `e2e_latency=3.7s`.
- The runtime path used SGLang + FlashInfer + our FlashQLA SM70/SM75 legacy
  backend + SM75 fallback patches.

What was patched or worked around:

- `ReqToTokenPool.write` Triton scatter
- vocab mask and LM head fallback paths
- mamba prefix handling
- decode allocations avoiding unsupported `clone`/`to` paths
- schedule batch add and `clamp_position`
- FlashInfer decode cumsum
- hybrid decode arange
- earlier FLA/GDN allocation and workspace issues

Why it is not production-ready:

- The smoke output was `ieee!`, not the expected `OK`.
- There is no valid prefill/decode benchmark.
- There is no 60-request serving run.
- The patch queue still needs to be exported, cleaned, and reproduced outside
  the experiment environment.

See:

- [BENCHMARKS.md](BENCHMARKS.md)
- [models/qwen3.6-27b-awq/vllm-mtp-k3.md](models/qwen3.6-27b-awq/vllm-mtp-k3.md)
- [models/qwen3.6-27b-awq/llamacpp-baseline.md](models/qwen3.6-27b-awq/llamacpp-baseline.md)
- [models/qwen3.6-27b-awq/sglang-smoke.md](models/qwen3.6-27b-awq/sglang-smoke.md)

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
