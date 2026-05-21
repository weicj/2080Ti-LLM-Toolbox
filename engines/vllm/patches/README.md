# vLLM Patch Queue

This directory records the small vLLM source changes that were required for the
validated dual RTX 2080 Ti route. It is intentionally a patch queue, not a
vendored vLLM checkout or virtualenv.

Apply patches from the root of a vLLM source tree, or from
`site-packages` inside a matching vLLM install:

```bash
patch -p0 < engines/vllm/patches/0001-sm75-flashqla-gdn-ragged-prefill.patch
```

## Current Patches

- `0001-sm75-flashqla-gdn-ragged-prefill.patch`

  Enables packed multi-prefill `cu_seqlens` batches for the FlashQLA
  SM70/SM75 legacy GDN path by splitting the packed batch per sequence, calling
  the legacy kernel, and reassembling output plus final state. This is the
  compatibility fix that made `max_num_seqs=4` serving usable on
  Qwen3.6-27B-AWQ.

## Overlay Inventory

The best tested vLLM build also included a TurboQuant KV overlay. That path is
not represented here as a tiny patch because it spans a full attention backend,
cache dtype registration, Triton store/decode ops, and TurboQuant quantization
helpers:

- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/v1/attention/ops/triton_turboquant_store.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`
- `vllm/model_executor/layers/quantization/turboquant/`
- TurboQuant cache dtype entries in vLLM config and backend registry files

The required behavior for the validated run is documented in
`../recipes/qwen36-27b-awq-turboquant-kv.md`: TurboQuant prefill must stay on
FlashInfer/FA2 where available, use the model attention scale, and SDPA must
remain a diagnostic fallback rather than the final route.

The file-level overlay inventory is pinned in
`../../../manifests/dual-2080ti-vllm-qwen27-awq-turboquant-overlay.lock`.
