# vLLM Recipe: Qwen3.6-27B-AWQ Best SM75 Route

Status: recommended current route.

This is the best validated dual RTX 2080 Ti vLLM route from the May 2026 lab
runs. It is the version to recreate first before trying TurboQuant KV or new
kernel experiments.

## Stack

```text
hardware = 2x modified RTX 2080 Ti 22GB, NVLink, SM75
engine = vLLM 0.21.0
torch = 2.11.0 cu130
flashinfer = 0.6.8
model = Qwen3.6-27B-AWQ
tensor_parallel_size = 2
quantization = awq_marlin
attention = FlashInfer full attention
gdn_prefill_backend = FlashQLA SM70/SM75 legacy
speculative = MTP K=3
```

Required local behavior:

- vLLM must select FlashQLA legacy for GDN prefill through
  `--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'`.
- Apply `../patches/0001-sm75-flashqla-gdn-ragged-prefill.patch` for
  concurrent serving. Without it, packed multi-prefill `cu_seqlens` batches hit
  the old single-contiguous-sequence limit.
- Keep FlashInfer/FlashAttention prefill active. SDPA fallback was diagnostic
  only and is not this route.
- Use the AWQ Marlin path for the AWQ checkpoint.

## Peak Single-Request Profile

This profile produced the highest documented 27B decode result:

```text
max_model_len = 8192
max_num_seqs = 1
max_num_batched_tokens = 8192
gpu_memory_utilization = 0.86
enable_chunked_prefill = true
async_scheduling = true
ignore_eos = true
speculative = {"method":"mtp","num_speculative_tokens":3}
```

Observed PP4096/TG128, measured rows only:

- median prefill: `1841.7 tok/s`
- median decode: `101.3 tok/s`
- max decode: `101.5 tok/s`
- median TTFT: `2.224s`
- median E2E: `3.485s`

Provenance: `reports/summaries/qwen36-27b-awq-vllm-peak-single-request.md`.

## Stable Serving Profile

This is the broader validated route used for 64K and repeated serving work:

```text
max_model_len = 65536
max_num_batched_tokens = 8192
max_num_seqs = 1
gpu_memory_utilization = 0.86
enable_chunked_prefill = true
speculative = {"method":"mtp","num_speculative_tokens":3}
cuda_graph_capture_sizes = [4]
```

Observed results:

- PP4096/TG128 earlier sweep: `1843.7 tok/s` prefill, `79.1 tok/s` decode,
  `3.8s` E2E.
- PP64K/TG512 cap: `1294.3 tok/s` prefill, `55.3 tok/s` decode, `56.8s` E2E.
- Sequential 60-request Ragent6 run: `167.4s` wall, average prefill
  `700.9 tok/s`, average generation `35.2 tok/s`.

## Concurrent Serving Profile

After the FlashQLA legacy GDN multi-prefill patch:

```text
max_num_seqs = 4
speculative = {"method":"mtp","num_speculative_tokens":3}
```

Validated behavior:

- Streaming PP3800/TG128 concurrency 1/2/4 completed.
- Ragent6 1/2/4-way shard functional checks completed with no GDN errors,
  tracebacks, or HTTP 500s.
- Quality stayed unchanged in the shard checks: strict `43/60`, partial
  weighted `82.5/100`, invalid `0`.

The GDN fix is a compatibility loop over packed sequences, not a fused ragged
GDN kernel. It makes serving usable; it is not the final performance shape for
multi-prefill batching.

## TurboQuant KV Branch

TurboQuant KV is a validated experimental branch of this route, not the default
best route yet. The successful TurboQuant rows used eager mode, no async
scheduling, `max_num_batched_tokens=4096`, MTP K=3, FlashQLA legacy GDN
prefill, and FlashInfer/FA2 prefill where available.

See `qwen36-27b-awq-turboquant-kv.md` for the separate KV dtype comparison:
`turboquant_4bit_nc` is the best practical TurboQuant row so far, while native
`int8_per_token_head` still reported much higher vLLM KV capacity in the current
experiment tree.
