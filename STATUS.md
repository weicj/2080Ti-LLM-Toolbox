# Status

## Recommended

### Qwen3.6-27B-AWQ + vLLM + FlashQLA + FlashInfer + MTP K=3

Status: best current route.

Why:

- Real 4K and 64K throughput numbers exist.
- A sequential 60-request real serving run completed in `167.39s`.
- Average prefill/generation throughput in that run was `700.9/35.2 tok/s`.
- Model quality was checked only as a sanity signal for the route.

## Experimental

### Qwen3.6-27B-AWQ + SGLang

Status: smoke-only.

The route reached a short `/generate` HTTP 200 on dual 2080 Ti, but generated
bad output (`ieee!`) and has no validated throughput or quality data.

### SGLang + Qwen3.6-27B GGUF

Status: blocked.

Earlier GGUF attempts hit dense tensor-parallel slicing and SM75 fused RMSNorm
issues before becoming serviceable.

## Not Recommended Yet

### zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ

Quality is decent, but vLLM only uses the slow AWQ path for this AutoRound AWQ
layout. Decode is about `4.65-5.27 tok/s`, which is not acceptable for the
target online use case.

### feanors/Qwen3.6-35B-A3B Claude/Opus distilled AWQ

Throughput is high, but Ragent6 quality collapsed: strict `10/60`, partial
weighted `32.9/100`. vLLM also reported many missing MoE expert weight warnings.

### cyankiwi/gemma-4-31B-it-AWQ-4bit

Loads to health in vLLM, then fails on the first 4K prefill because the forced
Triton attention path needs 96KB shared memory while SM75 exposes 64KB.
