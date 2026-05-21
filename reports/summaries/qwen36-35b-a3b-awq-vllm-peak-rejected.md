# Qwen3.6-35B-A3B AWQ vLLM High Throughput, Rejected Route

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2; one TU102-300A GPU and
one TU102-300 GPU.

Stack: vLLM 0.21, AWQ Marlin, FP16, FlashQLA SM70/SM75 legacy GDN prefill,
FlashInfer full attention, `max_model_len=65536`.

This derivative checkpoint reached near-100 TG, but it is not a recommended
route because the model quality collapsed in the follow-up Ragent6 run. The
validated 27B-AWQ route has a higher documented peak decode result.

## Throughput Sweep

| Route | PP4096/TG128 prefill | PP4096/TG128 decode | PP4096/TG128 E2E | PP64K/TG512 prefill | PP64K/TG512 decode | PP64K/TG512 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP off | `3235.3 tok/s` | `98.4 tok/s` | `2.57s` | `2748.0 tok/s` | `85.9 tok/s` | `29.25s` |
| MTP K=1 | `1263.2 tok/s` | `32.1 tok/s` | `7.24s` | `1809.5 tok/s` | `65.5 tok/s` | `43.19s` |
| MTP K=2 | `1324.0 tok/s` | `47.7 tok/s` | `5.78s` | `1805.6 tok/s` | `72.2 tok/s` | `42.54s` |
| MTP K=3 | `1334.2 tok/s` | `51.1 tok/s` | `5.57s` | `1801.4 tok/s` | `75.2 tok/s` | `42.34s` |
| MTP K=3 warmed 4K | `2004.5 tok/s` | `75.9 tok/s` | `3.73s` | - | - | - |

The near-100 TG result was the MTP-off PP4096/TG128 row: `98.4 tok/s` decode.
The MTP routes were slower for this model. For the current validated 27B-AWQ
peak, see
[qwen36-27b-awq-vllm-peak-single-request.md](qwen36-27b-awq-vllm-peak-single-request.md).

## Quality Result

Ragent6 0.2.2 zh-CN full60, native local serving:

- strict: `10/60`
- partial weighted: `32.9/100`
- partial raw: `21.44/60`
- invalid: `0`
- wall: `253.02s`

vLLM also logged many missing MoE expert weight warnings for
`down_proj.qweight/qzeros/scales`. The Ragent6 result suggests those warnings
were not harmless.

## Decision

Keep this as a high-throughput reference and failure note only. The recommended
dual-2080Ti vLLM route remains `Qwen3.6-27B-AWQ` with MTP K=3, and the
TurboQuant KV work should continue on that validated 27B route rather than this
35B-A3B checkpoint.

Provenance: private lab throughput and Ragent6 logs dated 2026-05-19, including
the `bench-qwen36-35b-a3b-awq-mtp0-20260519-003556.jsonl` throughput artifact
and the matching Ragent6 0.2.2 zh-CN full60 run.
