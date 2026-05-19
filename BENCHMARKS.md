# Benchmarks

All performance rows must keep prefill and decode separate. Aggregate tokens per
second are not enough for this repository.

## Qwen3.6-27B-AWQ

Hardware: miniclaw dual RTX 2080 Ti, tensor parallel size 2 for vLLM rows.

| Engine | Config | Prompt / Generate | Prefill | Decode | E2E | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vLLM | MTP off | 4K / 128 | `1858 tok/s` | `45.44 tok/s` | `5.05s` | Sweep baseline |
| vLLM | MTP K=1 | 4K / 128 | `1814.1 tok/s` | `47.08 tok/s` | `4.996s` | Small decode gain |
| vLLM | MTP K=2 | 4K / 128 | `1845.9 tok/s` | `63.42 tok/s` | `4.237s` | Good acceptance |
| vLLM | MTP K=3 | 4K / 128 | `1843.7 tok/s` | `79.14 tok/s` | `3.839s` | Best current 4K result |
| vLLM | MTP K=4 | 4K / 128 | `1856.0 tok/s` | `72.03 tok/s` | `3.984s` | Fourth token acceptance regressed |
| vLLM | MTP off | 64K / 512 | `1326 tok/s` | `36.22 tok/s` | `62.41s` | Sweep baseline |
| vLLM | MTP K=1 | 64K / cap | `1301.2 tok/s` | `48.30 tok/s` | `57.798s` | EOS before strict 512 |
| vLLM | MTP K=2 | 64K / cap | `1295.4 tok/s` | `53.04 tok/s` | `57.115s` | EOS before strict 512 |
| vLLM | MTP K=3 | 64K / cap | `1294.3 tok/s` | `55.33 tok/s` | `56.768s` | Best current 64K result, EOS at 405 tokens |
| llama.cpp | GGUF single 2080 Ti | 4114 / 128 | `553.38 tok/s` | `23.74 tok/s` | `12.84s` | T1 backend `18101`, `cache_prompt=false` |
| llama.cpp | GGUF single 2080 Ti | 64022 / 512 | `383.12 tok/s` | `16.29 tok/s` | `198.63s` | T1 backend `18101`, `cache_prompt=false` |

Ragent6 0.2.2 zh-CN, vLLM MTP K=3:

- strict: `43/60`
- partial weighted: `82.5/100`
- partial raw: `49.93/60`
- wall: `167.39s`
- prompt throughput median/mean/max: `867.8/700.9/1306.4 tok/s`
- generation throughput median/mean/max: `31.8/35.2/71.6 tok/s`
- speculative accepted/drafted: `4501/5616 = 80.15%`

## Other Models

| Model | Engine | Status | Prefill | Decode | Quality |
| --- | --- | --- | ---: | ---: | --- |
| zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ | vLLM AWQ slow path | Not recommended for online serving | `1242.7 tok/s` 4K | `5.27 tok/s` 4K | Ragent6 partial `80.4/100` |
| feanors/Qwen3.6-35B-A3B Claude/Opus distilled AWQ | vLLM AWQ Marlin | Not recommended due quality | `3235.3 tok/s` 4K | `98.4 tok/s` 4K | Ragent6 partial `32.9/100` |
| cyankiwi/gemma-4-31B-it-AWQ-4bit | vLLM compressed-tensors | Blocked on SM75 shared memory | - | - | No completed prefill |

## Source Pointers

- `/home/max/memory/2026-05-19.md`
- `/home/max/Develop/Ragent6/reports/qwen36_27b_awq_vllm_mtp_k3_ragent6_0_2_2_zh_CN_report_20260519_062013.md`
- `/home/max/reports/vllm_qwen36_35b_a3b_awq_bench_ragent6_20260519.md`
- `/home/max/reports/vllm_huihui_qwen36_27b_awq_bench_ragent6_20260519.md`
- `/home/max/reports/vllm_gemma4_31b_it_awq4_bench_ragent6_20260519.md`

