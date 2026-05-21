# Status

## Recommended

### Qwen3.6-27B-AWQ + vLLM + FlashQLA + FlashInfer + MTP K=3

Status: best current route.

Why:

- Real 4K and 64K throughput numbers exist.
- The current best-version recipe and required patch queue are recorded under
  `engines/vllm/recipes/qwen36-27b-awq-best-sm75.md` and
  `engines/vllm/patches/`.
- The current 4K single-request peak is PP4096/TG128 at `1841.7 tok/s` median
  prefill and `101.3 tok/s` median decode, with max decode `101.5 tok/s`.
- A sequential 60-request real serving run completed in `167.4s`.
- Average prefill/generation throughput in that run was `700.9/35.2 tok/s`.
- The updated vLLM build has validated `max_num_seqs=4` concurrent serving
  after the FlashQLA legacy GDN multi-prefill fix.
- Ragent6 1/2/4-way concurrent shard runs completed without GDN errors or HTTP
  500s, with unchanged quality: strict `43/60`, partial weighted `82.5/100`,
  invalid `0`. These shard runs are functional checks, not throughput
  benchmarks, because case runtimes are uneven.
- Model quality was checked only as a sanity signal for the route.

### Qwen3.6-27B-AWQ + vLLM + TurboQuant KV

Status: validated experimental SM75 compatibility.

Why:

- `turboquant_4bit_nc` and `turboquant_k8v4` both completed Ragent6 0.2.2
  zh-CN full60 with `invalid=0`.
- The successful route kept TurboQuant prefill on the FlashInfer/FA2 path and
  fixed the missing `sm_scale` argument in the TurboQuant ragged prefill plan.
- `tq4nc` is the stronger practical result so far: full60 wall `319s`,
  weighted `75.4`, invalid `0`, `max_model_len=43680`.
- `tqk8v4` completed full60 at `315s`, weighted `63.0`, invalid `0`, but needed
  `max_model_len=35840` in this run.
- Current experiment-tree resource behavior is not yet ideal: vLLM reported
  only `0.93 GiB` available KV memory for both TurboQuant rows, giving lower
  cache capacity than native INT8/FP16 rows despite the compressed KV format.

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
layout. Decode is about `4.7-5.3 tok/s`, which is not acceptable for the
target online use case.

### feanors/Qwen3.6-35B-A3B Claude/Opus distilled AWQ

This derivative checkpoint reached PP4096/TG128 `3235.3 tok/s` prefill and
`98.4 tok/s` decode with MTP off. It is still rejected because Ragent6 quality
collapsed: strict `10/60`, partial weighted `32.9/100`. vLLM also reported
many missing MoE expert weight warnings.

### cyankiwi/gemma-4-31B-it-AWQ-4bit

Loads to health in vLLM, then fails on the first 4K prefill because the forced
Triton attention path needs 96KB shared memory while SM75 exposes 64KB.
