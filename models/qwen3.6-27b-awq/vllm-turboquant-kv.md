# Qwen3.6-27B-AWQ: vLLM TurboQuant KV

Status: validated experimental SM75 compatibility.

Environment:

- dual modified RTX 2080 Ti 22GB, NVLink, TP=2
- vLLM 0.21 experiment tree
- torch 2.11 cu130
- flashinfer 0.6.8
- AWQ Marlin
- FlashQLA SM70/SM75 legacy GDN prefill
- TurboQuant prefill through FlashInfer/FA2
- MTP `num_speculative_tokens=3`
- eager mode, no async scheduling
- `max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.90`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

What changed:

- The successful route kept FA2 prefill active instead of using an SDPA
  fallback as the final path.
- The key TurboQuant backend fix was passing `sm_scale=self.scale` into
  `BatchPrefillWithRaggedKVCacheWrapper.plan()`.
- The launch wrapper had to correctly pass through `KV_CACHE_DTYPE`,
  `ASYNC_SCHEDULING`, and memory-related environment variables; earlier tests
  without that pass-through were not reliable A/B runs.

## Full60 KV dtype comparison

Workload: Ragent6 0.2.2 zh-CN full60, one active request at a time.

The prefill/decode columns are vLLM 10-second logger windows from uneven agent
requests, shown as `min/mean/max`; use wall time for the full-run comparison.

| KV dtype | Max model len | Available KV memory | GPU KV cache | Max concurrency | Prefill window | Decode window | Full60 wall | Weighted | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `turboquant_4bit_nc` | 43,680 | `0.93 GiB` | `58,800 tok` | `1.35x` | `0.0/354.6/952.9 tok/s` | `2.7/19.2/26.4 tok/s` | `319s` | `75.4` | `0` |
| `turboquant_k8v4` | 35,840 | `0.93 GiB` | `43,255 tok` | `1.21x` | `0.0/353.3/1145.5 tok/s` | `4.7/19.4/25.6 tok/s` | `315s` | `63.0` | `0` |
| `int8_per_token_head` | 43,680 | `7.47 GiB` | `312,312 tok` | `7.15x` | `18.1/384.5/952.9 tok/s` | `0.3/19.0/26.8 tok/s` | `296s` | `70.0` | `0` |
| `float16` | 43,680 | `7.47 GiB` | `187,106 tok` | `4.28x` | `23.9/471.4/1111.2 tok/s` | `3.5/20.1/29.2 tok/s` | `487s` | `74.8` | `0` |

Stability check:

- All four full60 rows completed `60/60` graded cases.
- Matching server logs had no `OutOfMemoryError`, `EngineDeadError`, traceback,
  HTTP 500, or connection reset.
- Earlier `tq4nc + MTP3` failures produced repeated special-token/garbage text;
  this did not recur after the FA2/sm_scale fix.

Interpretation:

- `turboquant_4bit_nc` is the useful TurboQuant KV result so far: it kept
  quality in the same band as FP16 while finishing much closer to INT8 wall
  time.
- `turboquant_k8v4` is stable but lower quality in this run and required a
  shorter `max_model_len=35840`.
- Native `int8_per_token_head` still had the best measured vLLM KV capacity in
  this experiment tree. TurboQuant compression is therefore not yet equivalent
  to better runtime capacity on SM75; allocator/workspace behavior still needs
  work.

## 250K KV footprint estimate

The table below is an allocator-derived estimate from the vLLM log lines
`Available KV cache memory` and `GPU KV cache size`. It is useful for comparing
resource slopes, but it is not proof that the current build can serve 250K in
that configuration.

| KV dtype | Log-derived bytes/token slope | Extrapolated KV footprint at 250K |
| --- | ---: | ---: |
| `turboquant_4bit_nc` | `~17 KiB/token` | `~4.0 GiB` |
| `turboquant_k8v4` | `~22 KiB/token` | `~5.4 GiB` |
| `int8_per_token_head` | `~25 KiB/token` | `~6.0 GiB` |
| `float16` | `~42 KiB/token` | `~10.0 GiB` |

The resource problem in this run was not the per-token slope alone. The
TurboQuant rows had much less vLLM-reported available KV memory after engine
initialization (`0.93 GiB`) than INT8/FP16 (`7.47 GiB`), which capped actual
cache size well below the extrapolated 250K target.

Evidence run IDs:

- `run-20260521-160048-tq4nc-fa2pure-eager-noasync-full-mb4096-u090-expseg`
- `run-20260521-161010-tqk8v4-fa2pure-eager-noasync-full-mb4096-u090-35840-expseg`
- `run-20260521-163518-awq-mtp3-int8-full-mb4096-u090`
- `run-20260521-164520-awq-mtp3-fp16-full-mb4096-u090`

Provenance: copied from private lab logs dated 2026-05-21.
