# 2080Ti LLM Toolbox

Recipes, patches, manifests, and benchmark summaries for running modern LLM
serving stacks on RTX 2080 Ti / Turing SM75.

This is a working toolbox, not an official support matrix. The main target is a
dual modified RTX 2080 Ti 22GB NVLink rig. The notes should still help other
SM75 users understand which paths are realistic and which ones are dead ends.

## Peak Result

Peak single-request speed on dual modified RTX 2080 Ti 22GB cards with NVLink,
using PP4096 / TG128:

| Model | Median Prefill | Median Decode |
| --- | ---: | ---: |
| `Qwen3.6-27B-AWQ` | `1841.7 tok/s` | `101.3 tok/s` |

That is the result to compare against when judging whether an SM75 setup is
actually in the right performance class. The detailed table and provenance are
in
[reports/summaries/qwen36-27b-awq-vllm-peak-single-request.md](reports/summaries/qwen36-27b-awq-vllm-peak-single-request.md).

## Recommended Route

This is the stack behind the peak result:

```text
vLLM 0.21.0
torch 2.11 cu130
TP=2
AWQ Marlin
FlashInfer/FA2 attention
FlashQLA SM70/SM75 legacy GDN prefill
MTP K=3
```

Use these first when reproducing the peak result or comparing other engines /
KV experiments:

- Recipe: [engines/vllm/recipes/qwen36-27b-awq-best-sm75.md](engines/vllm/recipes/qwen36-27b-awq-best-sm75.md)
- Patch queue: [engines/vllm/patches](engines/vllm/patches/README.md)
- Lock file: [manifests/dual-2080ti-vllm-qwen27-awq-best-sm75.lock](manifests/dual-2080ti-vllm-qwen27-awq-best-sm75.lock)
- Status page: [STATUS.md](STATUS.md)

Reproduction order:

1. Start from the best SM75 recipe.
2. Apply the vLLM patch queue.
3. Validate the MTP K=3 route before enabling TurboQuant KV.
4. Compare results against [BENCHMARKS.md](BENCHMARKS.md).

## Tested Hardware

The headline results are not from a stock 11GB card.

```text
GPU: 2x modified RTX 2080 Ti 22GB
Interconnect: NVLink
Architecture: Turing SM75
Stepping: TU102-300A + TU102-300
```

Single-request results mean one active request at a time. Concurrent serving
numbers are kept separate because they are not comparable with single-slot
llama.cpp rows.

## What Works

| Route | Status | Use It For |
| --- | --- | --- |
| vLLM + Qwen3.6-27B-AWQ + MTP K=3 | Recommended | Fast dual-card serving, 4K and 64K workloads, repeated agent-style requests |
| vLLM + TurboQuant KV | Validated experimental | Testing compressed KV behavior; `tq4nc` is the best practical TQ row so far |
| llama.cpp + Qwen3.6 27B GGUF | Reliable baseline | Simpler fallback and sanity checks |
| SGLang + Qwen3.6-27B-AWQ | Smoke-only | SM75 compatibility research, not production comparison |

The core SM75 unlock is FlashQLA for the GDN path. The separate FlashQLA backend
is maintained at [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75).

## Why This Is Recommended

The vLLM path has real performance, quality, and stability evidence:

- PP4096/TG128 peak repeat: see the peak table at the top of this README.
- PP64K/TG512: `1294.3 tok/s` prefill, `55.3 tok/s` decode.
- Sequential 60-request Ragent6 run: `167.4s` wall, average
  `700.9 tok/s` prefill and `35.2 tok/s` generation.
- Concurrent serving validated up to `max_num_seqs=4` after the FlashQLA legacy
  GDN multi-prefill patch.
- Ragent6 1/2/4-way shard checks completed without GDN errors, HTTP 500s, or
  quality regression: strict `43/60`, weighted `82.5/100`, invalid `0`.

The highest-quality detailed tables live in [BENCHMARKS.md](BENCHMARKS.md).
Curated run summaries live under [reports/summaries](reports/summaries/).

## Required vLLM Patch

The current vLLM build needs a FlashQLA legacy GDN compatibility fix for
multi-prefill batching:

- vLLM may pass multiple prefill sequences as packed `cu_seqlens`.
- The old FlashQLA legacy path only accepted one contiguous sequence.
- The patch splits the packed batch per sequence, calls the legacy GDN kernel,
  then reassembles output and final state.

Patch:
[engines/vllm/patches/0001-sm75-flashqla-gdn-ragged-prefill.patch](engines/vllm/patches/0001-sm75-flashqla-gdn-ragged-prefill.patch)

This is a compatibility loop, not a fused ragged GDN kernel. It makes real
serving usable; it is not the final multi-prefill performance design.

## TurboQuant KV Status

TurboQuant KV is useful, but not yet the default route.

Validated rows from the 2026-05-21 experiment tree:

| KV dtype | Estimated 256K KV footprint | Ragent6 Walltime | Ragent6 Result | Notes |
| --- | ---: | ---: | ---: | --- |
| `turboquant_4bit_nc` | `~4.2 GiB` | `319s` | `75.4` | Best practical TurboQuant result so far |
| `turboquant_k8v4` | `~5.7 GiB` | `315s` | `63.0` | Stable only at shorter context in this run |
| `int8_per_token_head` | `~6.3 GiB` | `296s` | `70.0` | Highest observed vLLM KV capacity in this build |
| `float16` | `~10.5 GiB` | `487s` | `74.8` | Slowest Ragent6 walltime |

Important caveat: the TurboQuant rows reported only `0.93 GiB` available vLLM KV
memory after engine initialization, lower than native INT8/FP16 rows in the same
experiment tree. Treat theoretical KV compression and current vLLM allocator
capacity as separate questions. The 256K KV footprints above are extrapolated
from vLLM cache-size logs, not validated 256K serving results.

Details:

- [engines/vllm/recipes/qwen36-27b-awq-turboquant-kv.md](engines/vllm/recipes/qwen36-27b-awq-turboquant-kv.md)
- [reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md](reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md)
- [manifests/dual-2080ti-vllm-qwen27-awq-turboquant-kv.lock](manifests/dual-2080ti-vllm-qwen27-awq-turboquant-kv.lock)
- [manifests/dual-2080ti-vllm-qwen27-awq-turboquant-overlay.lock](manifests/dual-2080ti-vllm-qwen27-awq-turboquant-overlay.lock)

## Baselines And Rejected Paths

llama.cpp remains the sanity baseline. It is slower than the validated vLLM
route on this hardware, but simpler and robust.

SGLang has reached a short HTTP 200 smoke with the Qwen3.6-27B-AWQ path, but it
is not production-ready: the smoke output was bad, and there is no valid
prefill/decode benchmark or repeated-request run yet.

The derivative `Qwen3.6-35B-A3B-AWQ` route reached near-100 tok/s decode, but is
rejected because quality collapsed in Ragent6: strict `10/60`, weighted
`32.9/100`.

Rejected and experimental routes are tracked in [STATUS.md](STATUS.md), model
notes under [models](models/README.md), and the relevant report summaries.

## Repository Map

- [engines/vllm](engines/vllm/README.md): vLLM SM75 recipes, patch queue, and launch notes.
- [engines/flashqla](engines/flashqla/README.md): FlashQLA SM70/SM75 kernel notes.
- [engines/llamacpp](engines/llamacpp/README.md): llama.cpp baselines.
- [engines/sglang](engines/sglang/README.md): SGLang experimental SM75 bring-up.
- [models](models/README.md): model-specific compatibility notes.
- [manifests](manifests/README.md): environment and workload locks.
- [reports](reports/index.md): curated summaries and raw log pointers.
- [BENCHMARKS.md](BENCHMARKS.md): full benchmark scorecard.
- [STATUS.md](STATUS.md): recommended, experimental, and rejected paths.
- [scripts](scripts/): small utilities for snapshots and repeatable calls.
