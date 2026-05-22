# Qwen3.6-27B-AWQ vLLM TurboQuant KV Summary

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2.

Stack: vLLM 0.21 experiment tree, torch 2.11 cu130, FlashInfer, FlashQLA
SM70/SM75 legacy GDN prefill, AWQ Marlin, MTP K=3, eager mode, no async
scheduling, `max_num_batched_tokens=4096`, `gpu_memory_utilization=0.90`.

Result:

- `turboquant_4bit_nc`: full60 wall `319s`, weighted `75.4`, invalid `0`,
  `max_model_len=43680`, vLLM KV cache `58,800 tokens`.
- `turboquant_k8v4`: full60 wall `315s`, weighted `63.0`, invalid `0`,
  `max_model_len=35840`, vLLM KV cache `43,255 tokens`.
- `int8_per_token_head`: full60 wall `296s`, weighted `70.0`, invalid `0`,
  vLLM KV cache `312,312 tokens`.
- `float16`: full60 wall `487s`, weighted `74.8`, invalid `0`,
  vLLM KV cache `187,106 tokens`.

The TurboQuant quality issue was fixed by staying on the FlashInfer/FA2 prefill
route and passing `sm_scale` into the TurboQuant ragged prefill plan. Earlier
bad runs produced repeated special-token and garbage output; the validated
full60 runs did not.

Full60 resource caveat:

The TurboQuant rows are validated for quality and stability, but not yet for
final resource efficiency. In the 2026-05-21 full60 `gpu_memory_utilization=0.90`
rows, vLLM reported only `0.93 GiB` available KV memory for both TurboQuant
rows, compared with `7.47 GiB` for the native INT8 and FP16 rows. That specific
quality-run capacity snapshot should not be used as the final long-context
capacity claim; use the 262K probe below for that.

262K startup/cache probe:

- `turboquant_4bit_nc`: READY, vLLM KV cache `735,084 tokens`,
  `20,595 MiB` total used VRAM per 2080 Ti.
- `turboquant_k8v4`: READY, vLLM KV cache `520,461 tokens`,
  `20,615 MiB` total used VRAM per 2080 Ti.
- `int8_per_token_head`: READY, vLLM KV cache `518,397 tokens`,
  `20,633 MiB` total used VRAM per 2080 Ti.
- `auto` / FP16: READY, vLLM KV cache `272,938 tokens`,
  `20,633 MiB` total used VRAM per 2080 Ti.

This is a real `max_model_len=262144` startup/cache/VRAM probe, not an
extrapolated footprint estimate. The VRAM numbers are total device memory after
startup, not KV-only footprint. It validates cache creation at 262K; it does not
by itself measure full 262K long-prompt throughput or quality.

Interpretation:

`tq4nc` is the best current TurboQuant option on this route. It is practical
enough to keep testing, but the next engineering target should be allocator and
workspace behavior, plus controlled PP/TG prefill/decode benchmarks outside the
Ragent6 full60 request stream.

Provenance: copied from private lab logs dated 2026-05-21.
