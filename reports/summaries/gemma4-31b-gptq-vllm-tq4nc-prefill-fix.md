# Gemma4 31B GPTQ vLLM tq4nc Prefill Fix

This summary records the 2026-05-24 Gemma4/Gemma31B TurboQuant KV prefill fix
on dual modified RTX 2080 Ti 22GB cards with vLLM TP=2.

The key conclusion is that the earlier Qwen3.6-27B TurboQuant performance route
was not lost. Qwen27 already used the successful SM75 TurboQuant path. The
Gemma4/Gemma31B slowdown was a separate `head_dim=512` final full-attention
problem.

## Tested Route

- Hardware: dual modified RTX 2080 Ti 22GB, TP=2.
- Model: `ebircak-gemma-4-31B-it-4bit-W4A16-GPTQ`.
- KV dtype: `turboquant_4bit_nc`.
- Workload: PP4096/TG128, no-MTP, single request, `ignore_eos=true`.
- Private lab artifact family: `g4assist-k-sweep-0523`.
- Private report pack: `gemma4_mtp_tq4nc_compat_20260523`.

## Root Cause

The old no-MTP `tq4nc` route was not simply slow because it used TurboQuant KV.
Torch profiler showed a final `kernel_unified_attention` launch at about
`728-748ms/rank` on SM75. The TurboQuant head512 layers were already handled by
the TQ SDPA fallback, but Gemma4's final standard full-attention layer still
fell back to the slow Triton/unified-attention path.

The fix is to pair both head512 fallbacks:

```bash
GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK=1
GEMMA4_SM75_SDPA_PREFILL512=1
```

The launcher now auto-enables `GEMMA4_SM75_SDPA_PREFILL512=1` when
`GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK=1` is set and the standard SDPA512
switch is unset.

## Result

Same-build PP4096/TG128 median comparison:

| Route | Median prefill | Median decode | Prefill ratio | Decode ratio |
| --- | ---: | ---: | ---: | ---: |
| non-TQ FP16 KV | `1693.30 tok/s` | `33.78 tok/s` | `100.0%` | `100.0%` |
| old tq4nc compile/cudagraph | `1248.70 tok/s` | `32.98 tok/s` | `73.7%` | `97.6%` |
| fixed tq4nc paired SDPA512 | `1646.56 tok/s` | `33.23 tok/s` | `97.2%` | `98.4%` |

The fixed row improves prefill by about `31.9%` over the old tq4nc route without
a decode regression.

## Boundary

This fixes Gemma4/Gemma31B no-MTP `tq4nc` prefill/decode parity for the current
PP4096/TG128 evaluator. It does not mean a true FlashInfer/FA2/flashattention-
turing `head_dim=512` fast path exists yet; the current solution is an SDPA/Torch
fallback that is fast enough for the measured lane.

`MTP3 + tq4nc` synthetic PP4096/TG128 remains a separate stability/performance
question and should not be used as the no-MTP `tq4nc` baseline.
