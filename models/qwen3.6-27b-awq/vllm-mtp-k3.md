# Qwen3.6-27B-AWQ: vLLM MTP K=3

Status: recommended current route, including validated concurrent serving.

Environment:

- dual modified RTX 2080 Ti 22GB, NVLink, TP=2
- one TU102-300A GPU plus one TU102-300 GPU
- vLLM 0.21
- torch 2.11 cu130
- flashinfer 0.6.8
- AWQ Marlin
- FlashInfer full attention
- FlashQLA SM70/SM75 legacy GDN prefill from
  [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)
- max model length `65536`
- MTP `num_speculative_tokens=3`
- `max_num_seqs=4` validated after the FlashQLA legacy GDN multi-prefill fix

Single-request performance:

| Workload | Prompt | Completion | Prefill | Decode | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/tg128, earlier sweep | 4096 | 128 | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` |
| 4K/tg128, peak repeat median | 4096 | 128 | `1841.7 tok/s` | `101.3 tok/s` | `3.5s` |
| 64K/tg512 cap | 64000 | 405 | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` |

The peak repeat used `max_model_len=8192`, `max_num_seqs=1`,
`max_num_batched_tokens=8192`, `gpu_memory_utilization=0.86`, chunked prefill,
async scheduling, MTP K=3, and `ignore_eos=true`. The three measured rows
excluding warmup were `101.3`, `98.4`, and `101.5 tok/s` decode, so the median
was `101.3 tok/s` and the max was `101.5 tok/s`.

Sequential 60-request serving run:

- workload: 60 real agent-style requests, one active request at a time
- wall: `167.4s`
- prompt throughput median/mean/max: `867.8/700.9/1306.4 tok/s`
- generation throughput median/mean/max: `31.8/35.2/71.6 tok/s`
- spec accepted/drafted: `4501/5616 = 80.2%`
- position acceptance mean: `0.927/0.854/0.782`
- quality sanity: strict `43/60`, partial weighted `82.5/100`

Compared with prior 27B llama.cpp runs, this was faster than the llama.cpp
baseline `471.0s` and llama.cpp MTP n=2 `306.0s` on the same Ragent6 line.

Concurrent serving validation:

| Workload | Concurrency | Prefill | Decode | E2E | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Streaming PP3800/TG128 | 1 | `1045.9 tok/s` | `35.2 tok/s` | `3.6s` | Per-request decode after TTFT was about `80.8 tok/s` |
| Streaming PP3800/TG128 x2 | 2 | `929.7 tok/s` | `31.3 tok/s` | `8.2s` | Total PP7600/TG256 |
| Streaming PP3800/TG128 x4 | 4 | `1318.3 tok/s` | `44.4 tok/s` | `11.5s` | Total PP15200/TG512 |

Ragent6 was also used as a functional concurrent check: 1/2/4-way shard runs all
completed without GDN errors, tracebacks, or HTTP 500s, and quality stayed at
strict `43/60`, partial weighted `82.5/100`, invalid `0`. Those shard runs are
not listed as throughput rows because the Ragent6 cases have uneven runtimes and
sharding changes ordering plus load balance.

The concurrency fix handles packed multi-sequence `cu_seqlens` by looping over
sequences for FlashQLA legacy GDN prefill and reassembling output/final state.
It is stable enough for `max_num_seqs=4`, but it is not the final high-performance
fused ragged GDN implementation.

Provenance: copied from private lab logs dated 2026-05-19 and 2026-05-20.
