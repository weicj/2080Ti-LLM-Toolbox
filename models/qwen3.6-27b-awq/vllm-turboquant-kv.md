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
- In the full60 `gpu_memory_utilization=0.90` rows, native
  `int8_per_token_head` had the best reported vLLM KV capacity. The later 262K
  startup/cache probe below uses a more aggressive capacity setting and shows
  the expected TurboQuant capacity advantage.

## 262K startup/cache probe

This later probe uses `max_model_len=262144`, `gpu_memory_utilization=0.98`,
`max_num_seqs=1`, `max_num_batched_tokens=4096`, eager mode, no async
scheduling, and MTP K=3. It validates startup, KV cache allocation, and VRAM
use, not full 262K long-prompt throughput.

| KV dtype | Startup | vLLM reported KV cache | Max concurrency at 262,144 | Total used VRAM / 2080 Ti |
| --- | --- | ---: | ---: | ---: |
| `turboquant_4bit_nc` | READY | `735,084 tok` | `2.80x` | `20,595 MiB` |
| `turboquant_k8v4` | READY | `520,461 tok` | `1.99x` | `20,615 MiB` |
| `int8_per_token_head` | READY | `518,397 tok` | `1.98x` | `20,633 MiB` |
| `auto` / FP16 | READY | `272,938 tok` | `1.04x` | `20,633 MiB` |

This replaces the earlier allocator-derived 250K/256K footprint estimates for
capacity claims. The VRAM column is total device memory after startup, including
weights, runtime/workspace, and KV cache; it is not KV-only footprint. The
biggest real startup/cache result is `turboquant_4bit_nc` at `735,084`
reported KV tokens.

Evidence run IDs:

- `run-20260521-160048-tq4nc-fa2pure-eager-noasync-full-mb4096-u090-expseg`
- `run-20260521-161010-tqk8v4-fa2pure-eager-noasync-full-mb4096-u090-35840-expseg`
- `run-20260521-163518-awq-mtp3-int8-full-mb4096-u090`
- `run-20260521-164520-awq-mtp3-fp16-full-mb4096-u090`

262K probe remote logs:

- FP16: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_fp16_161132-20260522-081144.log`
- INT8: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_int8_161541-20260522-081553.log`
- K8V4: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_k8v4_161834-20260522-081846.log`
- TQ4NC: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_k4v4_162222-20260522-082235.log`

Provenance: copied from private lab logs dated 2026-05-21.
