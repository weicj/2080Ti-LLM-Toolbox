# vLLM Recipe: Qwen3.6-27B-AWQ TurboQuant KV

Status: validated experimental SM75 compatibility.

Core settings used by the successful full60 runs:

```text
tensor_parallel_size = 2
quantization = awq_marlin
gdn_prefill_backend = flashqla_legacy
attention_prefill = FlashInfer/FA2
speculative = {"method":"mtp","num_speculative_tokens":3}
enforce_eager = true
async_scheduling = false
max_num_batched_tokens = 4096
max_num_seqs = 1
gpu_memory_utilization = 0.90
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

KV dtype settings:

```text
turboquant_4bit_nc:
  max_model_len = 43680

turboquant_k8v4:
  max_model_len = 35840
```

Launch argument sketch:

```bash
python -m vllm.entrypoints.openai.api_server \
  --dtype half \
  --tensor-parallel-size 2 \
  --quantization awq_marlin \
  --generation-config vllm \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enforce-eager \
  --no-async-scheduling \
  --kv-cache-dtype turboquant_4bit_nc \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Required code behavior:

- TurboQuant prefill should stay on the FlashInfer/FA2 path.
- `turboquant_attn.py` must pass `sm_scale=self.scale` when planning ragged
  prefill through FlashInfer.
- Do not describe SDPA fallback as the final route; it was diagnostic only.

Observed result:

- `turboquant_4bit_nc`: full60 wall `319s`, weighted `75.4`, invalid `0`.
- `turboquant_k8v4`: full60 wall `315s`, weighted `63.0`, invalid `0`.

Known caveat:

The current experiment tree reported only `0.93 GiB` available vLLM KV memory
for TurboQuant rows. This is enough for validated full60 runs, but it is not yet
the expected resource win from compressed KV.
