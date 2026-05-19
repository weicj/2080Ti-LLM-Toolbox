# Qwen3.6-27B-AWQ vLLM GDN Concurrency Summary

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2; one TU102-300A GPU and
one TU102-300 GPU.

Stack: vLLM 0.21, torch 2.11 cu130, FlashInfer attention, FlashQLA SM70/SM75
legacy GDN prefill, AWQ Marlin, MTP K=3, `max_num_seqs=4`.

Change:

- FlashQLA legacy GDN prefill previously only handled one contiguous prefill
  sequence.
- The current vLLM route handles packed multi-sequence `cu_seqlens` by looping
  over sequences, calling the legacy GDN kernel, then reassembling output and
  final state.
- This is a compatibility fix, not a fused ragged GDN kernel.

Streaming concurrency:

| Concurrency | Prompt / Generate | Prefill | Decode | E2E |
| --- | --- | ---: | ---: | ---: |
| 1 | PP3800/TG128 | `1045.9 tok/s` | `35.2 tok/s` | `3.6s` |
| 2 | PP7600/TG256 total | `929.7 tok/s` | `31.3 tok/s` | `8.2s` |
| 4 | PP15200/TG512 total | `1318.3 tok/s` | `44.4 tok/s` | `11.5s` |

Ragent6 60-request concurrent run:

| Concurrency | Prefill | Decode | E2E | Quality |
| ---: | ---: | ---: | ---: | --- |
| 1 | `770.5 tok/s` | `39.2 tok/s` | `164.0s` | strict `43/60`, invalid `0` |
| 2 | `815.1 tok/s` | `40.4 tok/s` | `151.0s` | strict `43/60`, invalid `0` |
| 4 | `944.2 tok/s` | `48.3 tok/s` | `124.0s` | strict `43/60`, invalid `0` |

Interpretation:

The vLLM route is no longer limited to single-request serving. Four concurrent
Ragent6 shards completed without GDN errors, tracebacks, or HTTP 500s, and model
quality stayed unchanged. Concurrency improves aggregate throughput, but TTFT
rises and per-request decode can drop, so these rows should stay separate from
single-request llama.cpp comparisons.
