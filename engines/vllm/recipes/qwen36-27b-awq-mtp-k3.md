# vLLM Recipe: Qwen3.6-27B-AWQ MTP K=3

Status: recommended current route.

Core settings:

```text
tensor_parallel_size = 2
max_model_len = 65536
max_num_batched_tokens = 8192
max_num_seqs = 1
gpu_memory_utilization = 0.86
quantization = awq_marlin
gdn_prefill_backend = flashqla_legacy
attention = FlashInfer
speculative = {"method":"mtp","num_speculative_tokens":3}
cuda_graph_capture_sizes = [4]
```

Observed result:

- 4K/tg128: `1843.7 tok/s` prefill, `79.1 tok/s` decode, `3.8s` e2e.
- 64K/tg512 cap: `1294.3 tok/s` prefill, `55.3 tok/s` decode, `56.8s` e2e.
- Sequential 60-request serving run: `167.4s` wall, average prefill
  `700.9 tok/s`, average generation `35.2 tok/s`.

TODO: export the exact launch script from the private lab experiment and remove
machine-specific paths.
