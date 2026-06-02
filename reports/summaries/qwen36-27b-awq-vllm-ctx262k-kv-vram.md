# Qwen3.6-27B-AWQ vLLM 262K KV/VRAM and Real-Prompt Probe

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
`max_model_len=262144`. The startup/cache table does not by itself measure full
262K long-prompt throughput or quality; the real-prompt gates below are the
stronger evidence for actual native-context serving.

## Real 262K Prompt Gates

These rows used real requests with `prompt_tokens=262000`, not only engine
startup. They are single-request gates with one generated token, so decode is
not a meaningful throughput signal except as a basic completion sanity check.

| Route | Model | KV dtype | MTP | Request | Prefill | Decode note | Result |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| original AWQ baseline | `QuantTrio-Qwen3.6-27B-AWQ` | FP16/default | off | PP262000/TG1 | `785.26 tok/s` | one-token decode only | HTTP 200 |
| current Qwen quality lane | `mconcat/Qwopus3.6-27B-v2-AWQ-4bit` | `turboquant_4bit_nc` | off | PP262000/TG1 | `783.09 tok/s` | one-token decode only | HTTP 200 |

The Qwopus TQ4NC gate required the continuation workspace reserve to cover the
full native context. Smaller reserves passed shorter prompts but failed near
262K after the workspace was locked. A separate two-request gate with
`gpu_memory_utilization=0.88`, `max_num_seqs=2`, and the same 262K reserve
completed two PP262000/TG1 requests in one server run, but logs showed the
prefills were serialized; that proves queued capacity, not parallel long-prefill
throughput.

Private lab artifact labels:

- FP16: `qwen27_ctx262k_fp16_20260522`
- INT8: `qwen27_ctx262k_int8_20260522`
- K8V4: `qwen27_ctx262k_k8v4_20260522`
- TQ4NC: `qwen27_ctx262k_tq4nc_20260522`

Real-prompt result pointers:

- QuantTrio FP16/default KV: `quanttrio_fp16_nomtp_pp262000_tg1_20260602`
- Qwopus TQ4NC: `qwen_tq4nc_nomtp_ctx262k_longprompt_gate_1780285496`
- Qwopus TQ4NC two-request gate:
  `qwen_tq4nc_nomtp_ctx262k_longprompt_gate_1780287143`

Provenance: copied from private lab logs dated 2026-05-22 through 2026-06-02.
