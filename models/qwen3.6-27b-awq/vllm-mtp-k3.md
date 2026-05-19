# Qwen3.6-27B-AWQ: vLLM MTP K=3

Status: recommended current route.

Environment:

- miniclaw
- dual RTX 2080 Ti, TP=2
- vLLM 0.21
- torch 2.11 cu130
- flashinfer 0.6.8
- AWQ Marlin
- FlashInfer full attention
- FlashQLA legacy GDN prefill
- max model length `65536`
- MTP `num_speculative_tokens=3`

Performance:

| Workload | Prompt | Completion | Prefill | Decode | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K/tg128 | 4096 | 128 | `1843.7 tok/s` | `79.14 tok/s` | `3.839s` |
| 64K/tg512 cap | 64000 | 405 | `1294.3 tok/s` | `55.33 tok/s` | `56.768s` |

Ragent6 0.2.2 zh-CN:

- strict: `43/60`
- partial weighted: `82.5/100`
- partial raw: `49.93/60`
- invalid: `0`
- wall: `167.39s`
- spec accepted/drafted: `4501/5616 = 80.15%`
- position acceptance mean: `0.927/0.854/0.782`

Compared with prior 27B llama.cpp runs, this was faster than the llama.cpp
baseline `471s` and llama.cpp MTP n=2 `306s` on the same Ragent6 line.

Source pointers:

- `/home/max/memory/2026-05-19.md`
- `/home/max/Develop/Ragent6/reports/qwen36_27b_awq_vllm_mtp_k3_ragent6_0_2_2_zh_CN_report_20260519_062013.md`

