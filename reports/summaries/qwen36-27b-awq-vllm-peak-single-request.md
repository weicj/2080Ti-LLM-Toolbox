# Qwen3.6-27B-AWQ vLLM Peak Single-Request Decode

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2; one TU102-300A GPU and
one TU102-300 GPU.

Stack: vLLM 0.21, torch 2.11 cu130, AWQ Marlin, FlashInfer attention, FlashQLA
SM70/SM75 legacy GDN prefill, MTP K=3, chunked prefill, async scheduling,
`max_model_len=8192`, `max_num_seqs=1`, `max_num_batched_tokens=8192`,
`gpu_memory_utilization=0.86`, `ignore_eos=true`.

This is the current highest documented dual-2080Ti vLLM decode/TG result for
the validated `Qwen3.6-27B-AWQ` route.

## PP4096/TG128 Repeat

| Row | Prefill | Decode | TTFT | E2E |
| --- | ---: | ---: | ---: | ---: |
| warmup | `1599.4 tok/s` | `70.1 tok/s` | `2.561s` | `4.388s` |
| measure1 | `1853.7 tok/s` | `101.3 tok/s` | `2.210s` | `3.473s` |
| measure2 | `1840.1 tok/s` | `98.4 tok/s` | `2.226s` | `3.527s` |
| measure3 | `1841.7 tok/s` | `101.5 tok/s` | `2.224s` | `3.485s` |

Measured rows only, excluding warmup:

- median prefill: `1841.7 tok/s`
- median decode: `101.3 tok/s`
- max decode: `101.5 tok/s`
- median TTFT: `2.224s`
- median E2E: `3.485s`

## Why It Matters

The older PP4096/TG128 scorecard row for the same model and MTP K=3 recorded
`1843.7 tok/s` prefill and `79.1 tok/s` decode. The later single-request
INT4-goal repeat restored the 4K prefill level and pushed decode to about
`100 tok/s` on the same validated 27B-AWQ route.

This is still a single-request microbenchmark. It should not be mixed with
multi-concurrency aggregate throughput or Ragent6 10-second logger windows.

## Provenance

Private lab logs dated 2026-05-19:

- result artifact: `results/int4-goal-single-pp4096-tg128-multimeasure.jsonl`
- server log: `vllm-qwen36-27b-awq-int4-multimeasure-20260519-231226.log`

The matching server log confirms `awq_marlin`, `MarlinLinearKernel`,
FlashQLA legacy GDN prefill, FlashInfer sampling, MTP K=3, `max_model_len=8192`,
`max_num_seqs=1`, and `max_num_batched_tokens=8192`.
