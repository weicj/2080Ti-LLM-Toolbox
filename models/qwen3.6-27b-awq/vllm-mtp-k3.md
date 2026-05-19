# Qwen3.6-27B-AWQ: vLLM MTP K=3

Status: recommended current route.

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

Single-request performance:

| Workload | Prompt | Completion | Prefill | Decode | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/tg128 | 4096 | 128 | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` |
| 64K/tg512 cap | 64000 | 405 | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` |

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

Provenance: copied from private lab logs dated 2026-05-19.
