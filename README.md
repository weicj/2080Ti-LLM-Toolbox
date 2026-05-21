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
| vLLM | Recommended route | `Qwen3.6-27B-AWQ` | Fast single-request serving and validated multi-request serving on dual 22GB 2080 Ti with TP=2, FlashInfer, FlashQLA, AWQ Marlin, and MTP K=3; TurboQuant KV is now validated as an experimental KV route |
| llama.cpp | Reliable baseline | upstream-original Qwen3.6 27B GGUF conversion | Practical single-card baseline and fallback; slower long-prefill than vLLM, but simple and robust |
| SGLang | Experimental | `Qwen3.6-27B-AWQ` | Compatibility research and patch archive; not ready for production comparison |

FlashQLA is the kernel foundation for the vLLM and SGLang GDN path here. The
SM70/SM75 backend is maintained separately at
[weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75).
The current vLLM build also includes a GDN multi-prefill compatibility fix for
FlashQLA legacy: packed `cu_seqlens` prefill batches are split per sequence and
reassembled. That makes `max_num_seqs=4` usable on the validated Qwen3.6 27B
route, but it is still a compatibility loop rather than a fused ragged GDN
kernel.

The 2026-05-21 experiment tree also validated TurboQuant KV on the same dual
2080 Ti route. `turboquant_4bit_nc` and `turboquant_k8v4` both completed the
Ragent6 0.2.2 zh-CN full60 run with `invalid=0` after keeping the FA2 prefill
route and passing `sm_scale` into the TurboQuant FlashInfer ragged prefill plan.
This is recorded as validated experimental SM75 compatibility, not as the
default stable route yet.

## Qwen3.6 27B Artifacts

The tables below only use Qwen3.6 27B-family artifacts. Derivative model lines
are intentionally excluded from these comparisons.

| Artifact | Route | Size / Quantization | Notes |
| --- | --- | --- | --- |
| `QuantTrio-Qwen3.6-27B-AWQ` | vLLM / SGLang | 21.9 GB safetensors, AWQ 4-bit, group size 128, zero point enabled | Includes MTP tensors; vLLM validated with AWQ Marlin |
| `unsloth/Qwen3.6-27B-GGUF` `Qwen3.6-27B-Q4_K_M.gguf` | llama.cpp baseline | 16.8 GB file, GGUF `Q4_K_M` | Upstream-original Qwen3.6 conversion used for llama.cpp baseline rows |
| RDson `Qwen3.6-27B-MTP-Q4_K_M.gguf` | llama.cpp integrated MTP | 16.5 GB file, GGUF `Q4_K_M` with integrated MTP tensors | Separate artifact; used only for llama.cpp MTP rows |

## Mature Framework Comparison

The table below only compares routes with usable performance data. SGLang is
tracked separately because the current work is still compatibility bring-up, not
a production-serving result.

Single-request serving measurements grouped by workload:

### PP4096 / TG128

| Framework | Route | Model | Prefill | Decode | E2E | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | TP=2, MTP off | Qwen3.6-27B-AWQ | `1858.0 tok/s` | `45.4 tok/s` | `5.1s` | Dual-card baseline |
| vLLM | TP=2, MTP K=3 | Qwen3.6-27B-AWQ | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` | Earlier sweep row |
| vLLM | TP=2, MTP K=3 peak repeat | Qwen3.6-27B-AWQ | `1841.7 tok/s` | `101.3 tok/s` | `3.5s` | Current peak 27B route; max decode `101.5 tok/s` |
| llama.cpp | upstream-original GGUF baseline | Qwen3.6-27B-Q4_K_M | `553.4 tok/s` | `23.7 tok/s` | `12.8s` | Single-card baseline |

### PP64K / TG512

| Framework | Route | Model | Prefill | Decode | E2E | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | TP=2, MTP off | Qwen3.6-27B-AWQ | `1326.0 tok/s` | `36.2 tok/s` | `62.4s` | Dual-card baseline |
| vLLM | TP=2, MTP K=3 | Qwen3.6-27B-AWQ | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` | Best current long-context route |
| llama.cpp | upstream-original GGUF baseline | Qwen3.6-27B-Q4_K_M | `383.1 tok/s` | `16.3 tok/s` | `198.6s` | Single-card baseline |

### PP16K / TG4096

This workload records the llama.cpp integrated-MTP GGUF route. It is not mixed
into the PP4096/TG128 or PP64K/TG512 tables because it used a different prompt
and generation length.

| Framework | Route | Model | Prefill | Decode | E2E | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| llama.cpp | baseline, same MTP GGUF artifact | RDson Qwen3.6-27B-MTP-Q4_K_M | `609.4 tok/s` | `18.5 tok/s` | `248.9s` | MTP disabled |
| llama.cpp | integrated MTP n=2 | RDson Qwen3.6-27B-MTP-Q4_K_M | `501.3 tok/s` | `28.4 tok/s` | `177.0s` | `68.0%` draft acceptance |
| llama.cpp | integrated MTP n=3 | RDson Qwen3.6-27B-MTP-Q4_K_M | `496.8 tok/s` | `27.7 tok/s` | `181.2s` | `60.7%` draft acceptance |

Interpretation:

- vLLM is the current lead for the dual-card 22GB setup, especially long-context
  prefill and MTP-assisted decode.
- llama.cpp is still the practical baseline and the sanity check for every
  optimization claim.

Peak-throughput note: the highest documented vLLM decode/TG number for the
validated 27B route is `101.5 tok/s` max, `101.3 tok/s` median, from a
PP4096/TG128 MTP K=3 repeat with `max_model_len=8192`, `max_num_seqs=1`, and
`max_num_batched_tokens=8192`. A derivative `Qwen3.6-35B-A3B-AWQ` checkpoint
also reached `98.4 tok/s`, but that route is not recommended: Ragent6 0.2.2
zh-CN collapsed to strict `10/60` and partial weighted `32.9/100`. See
[reports/summaries/qwen36-27b-awq-vllm-peak-single-request.md](reports/summaries/qwen36-27b-awq-vllm-peak-single-request.md),
[models/qwen3.6-35b-a3b-awq](models/qwen3.6-35b-a3b-awq/README.md), and
[reports/summaries/qwen36-35b-a3b-awq-vllm-peak-rejected.md](reports/summaries/qwen36-35b-a3b-awq-vllm-peak-rejected.md).

## Sequential 60-Request Serving Run

For agent-style workload testing, this repository uses a 60-request sequential
run: real requests are sent one after another to the same server. This measures
single-request serving behavior under repeated use without hiding latency inside
concurrent batching. The workload is from
[Ragent6](https://github.com/weicj/Ragent6), using its 60-case agent benchmark as
a repeatable request stream.

| Framework | Route | Model | Prefill | Decode | E2E | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | TP=2, MTP K=2 | Qwen3.6-27B-AWQ | `631.9 tok/s` | `31.3 tok/s` | `190.6s` | Slower than K=3 despite higher acceptance |
| vLLM | TP=2, MTP K=3 | Qwen3.6-27B-AWQ | `700.9 tok/s` | `35.2 tok/s` | `167.4s` | Current best validated route |
| llama.cpp | baseline, same MTP GGUF artifact | RDson Qwen3.6-27B-MTP-Q4_K_M | `350.3 tok/s` | `21.2 tok/s` | `471.0s` | MTP disabled |
| llama.cpp | integrated MTP n=2 | RDson Qwen3.6-27B-MTP-Q4_K_M | `297.0 tok/s` | `45.1 tok/s` | `306.0s` | Faster decode, prefill penalty |

Model scores from these runs are treated as sanity checks only. This repository
is about whether the 2080 Ti serving stack runs fast and correctly; benchmark
scores mostly reflect the model.

### KV dtype comparison under the 2026-05-21 experiment tree

These rows use the same Ragent6 0.2.2 zh-CN full60 request stream with vLLM
TP=2, AWQ Marlin, FlashQLA legacy GDN prefill, FlashInfer/FA2 TurboQuant
prefill where applicable, MTP K=3, eager mode, no async scheduling,
`max_num_batched_tokens=4096`, and `gpu_memory_utilization=0.90`.

The prefill/decode columns here are vLLM 10-second logger windows from uneven
agent requests, shown as `min/mean/max`; use wall time for the full-run
comparison.

| KV dtype | Max model len | Available KV memory | GPU KV cache | Max concurrency | Prefill window | Decode window | Full60 wall | Weighted | Invalid | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `turboquant_4bit_nc` | 43,680 | `0.93 GiB` | `58,800 tok` | `1.35x` | `0.0/354.6/952.9 tok/s` | `2.7/19.2/26.4 tok/s` | `319s` | `75.4` | `0` | Validated, best TQ quality |
| `turboquant_k8v4` | 35,840 | `0.93 GiB` | `43,255 tok` | `1.21x` | `0.0/353.3/1145.5 tok/s` | `4.7/19.4/25.6 tok/s` | `315s` | `63.0` | `0` | Validated, shorter context |
| `int8_per_token_head` | 43,680 | `7.47 GiB` | `312,312 tok` | `7.15x` | `18.1/384.5/952.9 tok/s` | `0.3/19.0/26.8 tok/s` | `296s` | `70.0` | `0` | Capacity leader in this build |
| `float16` | 43,680 | `7.47 GiB` | `187,106 tok` | `4.28x` | `23.9/471.4/1111.2 tok/s` | `3.5/20.1/29.2 tok/s` | `487s` | `74.8` | `0` | Slowest full60 wall |

Interpretation:

- `tq4nc` is the practical TurboQuant result from this run: quality matched the
  FP16 sanity band while keeping wall time close to INT8.
- Current TurboQuant allocator behavior in this experiment tree left only
  `0.93 GiB` as vLLM-reported available KV memory, so the measured cache
  capacity is lower than native `int8_per_token_head` or `float16` despite the
  compressed KV format. Treat theoretical KV compression and current vLLM
  allocator capacity as separate resource questions.
- `tqk8v4` was stable only at `max_model_len=35840` in this full60 run; the
  `43680` setting failed earlier during KV capacity initialization.

## Concurrent Serving Validation

These rows are intentionally separate from the single-request comparison above.
They measure vLLM batching behavior after the FlashQLA GDN multi-prefill fix.
The route is Qwen3.6-27B-AWQ, TP=2, MTP K=3, `max_num_seqs=4`, FlashInfer
attention, AWQ Marlin, and FlashQLA SM70/SM75 legacy GDN prefill.

### Streaming PP3800 / TG128

| Concurrency | Prompt / Generate | Prefill | Decode | E2E | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | PP3800/TG128 | `1045.9 tok/s` | `35.2 tok/s` | `3.6s` | Per-request decode after TTFT was about `80.8 tok/s` |
| 2 | PP7600/TG256 total | `929.7 tok/s` | `31.3 tok/s` | `8.2s` | Two simultaneous streaming requests |
| 4 | PP15200/TG512 total | `1318.3 tok/s` | `44.4 tok/s` | `11.5s` | Four simultaneous streaming requests |

### Ragent6 Concurrent Functional Check

This uses [Ragent6](https://github.com/weicj/Ragent6) as the request stream,
split into concurrent shards. This is a functional validation, not a clean
throughput benchmark: Ragent6 cases have uneven runtimes, so splitting the suite
into shards changes ordering and load balance. The useful result is that 1/2/4
concurrency completed without GDN errors, tracebacks, or HTTP 500s, and quality
was unchanged: strict `43/60`, partial weighted `82.5/100`, invalid `0`.

Interpretation:

- The new vLLM build is no longer single-request-only for this route.
- Use the streaming PP3800/TG128 rows for controlled multi-request throughput.
- Use the Ragent6 concurrent check only as evidence that real agent-style
  concurrent requests stay stable and preserve quality.

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
- [models/qwen3.6-27b-awq/vllm-turboquant-kv.md](models/qwen3.6-27b-awq/vllm-turboquant-kv.md)
- [reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md](reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md)
- [reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md](reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md)
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
