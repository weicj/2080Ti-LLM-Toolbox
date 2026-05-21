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

Resource caveat:

The TurboQuant rows are validated for quality and stability, but not yet for
final resource efficiency. In this experiment tree, vLLM reported only
`0.93 GiB` available KV memory for both TurboQuant rows, compared with
`7.47 GiB` for the native INT8 and FP16 rows. This makes measured cache capacity
lower than native INT8/FP16 even though the KV format is compressed.

250K allocator-derived extrapolation:

- `turboquant_4bit_nc`: about `4.0 GiB` KV footprint at 250K.
- `turboquant_k8v4`: about `5.4 GiB` KV footprint at 250K.
- `int8_per_token_head`: about `6.0 GiB` KV footprint at 250K.
- `float16`: about `10.0 GiB` KV footprint at 250K.

This is a slope comparison derived from current vLLM cache-size logs, not a
validated 250K serving result.

Interpretation:

`tq4nc` is the best current TurboQuant option on this route. It is practical
enough to keep testing, but the next engineering target should be allocator and
workspace behavior, plus controlled PP/TG prefill/decode benchmarks outside the
Ragent6 full60 request stream.

Provenance: copied from private lab logs dated 2026-05-21.
