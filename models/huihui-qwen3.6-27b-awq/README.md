# Huihui Qwen3.6-27B Abliterated AWQ

Model: `zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ`

Status: not recommended for online serving on the tested vLLM path.

Reason:

- The weight is AutoRound AWQ `version=gemm`, `zero_point=false`.
- vLLM could only use the slow `awq` path.
- Forcing `awq_marlin` failed because Marlin did not support the required uint4
  scalar type.

Throughput:

| Config | Workload | Prefill | Decode | E2E |
| --- | --- | ---: | ---: | ---: |
| MTP off, vLLM AWQ slow path | 4K/tg128 | `1242.7 tok/s` | `5.27 tok/s` | `27.58s` |
| MTP off, vLLM AWQ slow path | 64K/tg512 | `1243.1 tok/s` | `4.65 tok/s` | `161.60s` |
| K=1 | 4K/tg128 | `1198.3 tok/s` | `5.13 tok/s` | `28.37s` |
| K=1 | 64K/tg512 | OOM | OOM | OOM |

Quality:

- Ragent6 0.2.2 zh-CN strict `40/60`
- partial weighted `80.4/100`

Conclusion: quality is decent, but decode speed is too slow for this toolbox's
target use case.

