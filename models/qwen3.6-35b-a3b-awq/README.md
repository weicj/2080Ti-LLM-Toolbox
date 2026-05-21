# Qwen3.6-35B-A3B AWQ

Model:
`feanors/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4`

Status: not recommended despite near-100 dual-2080Ti vLLM decode/TG.

Runtime:

- dual modified RTX 2080 Ti 22GB, NVLink
- TP=2
- vLLM 0.21
- AWQ Marlin
- FP16
- max model length `65536`
- FlashQLA SM70/SM75 legacy GDN prefill
- FlashInfer full attention

Throughput, MTP off:

| Workload | Prefill | Decode | E2E |
| --- | ---: | ---: | ---: |
| pp4096/tg128 | `3235.3 tok/s` | `98.4 tok/s` | `2.6s` |
| pp64K/tg512 | `2748.0 tok/s` | `85.9 tok/s` | `29.3s` |

The near-100 TG result was the MTP-off `pp4096/tg128` row: `98.4 tok/s`
decode. MTP did not help this checkpoint; the K=3 row was only `51.1 tok/s`
decode on the same `pp4096/tg128` sweep.

Quality:

- Ragent6 0.2.2 zh-CN strict `10/60`
- partial weighted `32.9/100`

Risk:

vLLM logged many missing MoE expert weight warnings for
`down_proj.qweight/qzeros/scales`. The quality result suggests these warnings
were not harmless.

See also:

- [../../reports/summaries/qwen36-35b-a3b-awq-vllm-peak-rejected.md](../../reports/summaries/qwen36-35b-a3b-awq-vllm-peak-rejected.md)
