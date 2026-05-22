# Qwen3.6-27B-AWQ vLLM 262K KV/VRAM Probe

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2.

Stack: vLLM 0.21 SM75 experiment tree, torch 2.11 cu130, FlashInfer/FA2,
FlashQLA SM70/SM75 legacy GDN prefill, AWQ Marlin, MTP K=3.

Probe settings:

- `max_model_len=262144`
- `gpu_memory_utilization=0.98`
- `max_num_seqs=1`
- `max_num_batched_tokens=4096`
- eager mode, no async scheduling
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728`

## Result

| KV dtype | Startup | vLLM reported KV cache | Max concurrency at 262,144 | Total used VRAM / 2080 Ti |
| --- | --- | ---: | ---: | ---: |
| `auto` / FP16 | READY | `272,938 tok` | `1.04x` | `20,633 MiB` |
| `int8_per_token_head` | READY | `518,397 tok` | `1.98x` | `20,633 MiB` |
| `turboquant_k8v4` | READY | `520,461 tok` | `1.99x` | `20,615 MiB` |
| `turboquant_4bit_nc` | READY | `735,084 tok` | `2.80x` | `20,595 MiB` |

## Interpretation

This probe replaces the earlier allocator-derived 250K/256K footprint estimate
for capacity claims. The VRAM column is total `nvidia-smi` device memory after
startup, including weights, runtime/workspace, and KV cache; it is not KV-only
footprint. Under the most aggressive tested VRAM allocation, FP16 can create a
262K-context vLLM cache with `272,938` reported KV tokens. Native INT8 roughly
doubles that to `518,397` tokens. `turboquant_k8v4` is essentially the same
capacity as native INT8 in this startup probe, while `turboquant_4bit_nc` is
the largest measured row at `735,084` tokens.

This validates engine startup, KV cache allocation, and VRAM use at
`max_model_len=262144`. It does not by itself measure full 262K long-prompt
throughput or quality.

Remote log pointers:

- FP16: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_fp16_161132-20260522-081144.log`
- INT8: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_int8_161541-20260522-081553.log`
- K8V4: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_k8v4_161834-20260522-081846.log`
- TQ4NC: `/data/experiments/vllm-qwen27-awq-sm75-fa-turing-prefill-20260520/vllm-qwen27_ctx262k_k4v4_162222-20260522-082235.log`

Provenance: copied from private lab logs dated 2026-05-22.
