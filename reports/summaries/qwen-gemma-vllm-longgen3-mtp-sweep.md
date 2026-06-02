# Qwen/Gemma vLLM LongGen3 MTP Sweep

This summary records the 2026-06-02 LongGen3 Chinese 4096/1024 warm sweep on
dual modified RTX 2080 Ti 22GB cards with vLLM TP=2.

The goal was to measure speculative decoding on a more realistic long-output
workload than PP4096/TG128. Each model/K run used three non-scored warmups:
short service warmup, long-prefill warmup, and 256-token decode warmup. The
scored rows exclude service startup, model loading, torch.compile/AOT,
CUDAGraph capture, FlashInfer/Triton JIT, and first-request JIT.

## Runtime

- Hardware: dual modified RTX 2080 Ti 22GB, TP=2.
- Runtime: vLLM SM75 CUDA 12.8 stable tree.
- KV dtype: FP16/default KV.
- Execution: no-eager/CUDAGraph.
- Qwen route: Qwopus AWQ, native MTP, Qwopus BF16 draft path, FlashQLA legacy
  GDN prefill.
- Gemma route: Gemma4 31B GPTQ target, external assistant MTP, Gemma text-only
  FlashInfer override, SDPA512 fallback for head_dim 512.

## Qwen/Qwopus27 AWQ

Scored total per K: `9836` prompt tokens and `3072` completion tokens.

| MTP | Scored wall | Prefill | Decode | Completion | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| off | `95.217s` | `103.30 tok/s` | `32.26 tok/s` | `3072` | Baseline |
| K=1 | `69.351s` | `141.83 tok/s` | `44.30 tok/s` | `3072` | Clear gain |
| K=2 | `62.074s` | `158.46 tok/s` | `49.49 tok/s` | `3072` | Clear gain |
| K=3 | `56.743s` | `173.34 tok/s` | `54.14 tok/s` | `3072` | Conservative deployment point |
| K=4 | `55.838s` | `176.15 tok/s` | `55.02 tok/s` | `3072` | Near plateau |
| K=5 | `55.654s` | `176.73 tok/s` | `55.20 tok/s` | `3072` | Numeric best in this sweep |

Interpretation: MTP is clearly effective on this 4096/1024 long-output
workload. K=5 is the numeric best, but K=3 through K=5 are close enough that
K=3 remains the conservative deployment reference when acceptance and output
quality risk matter more than the last few percent.

## Gemma4 31B GPTQ

Scored total per K: `10686` prompt tokens. Completion tokens vary because the
prose case often stops before the 1024-token cap.

| MTP | Scored wall | Prefill | Decode | Completion | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| off | `100.171s` | `106.68 tok/s` | `30.67 tok/s` | `3072` | Baseline |
| K=1 | `142.019s` | `75.24 tok/s` | `21.06 tok/s` | `2991` | Regression |
| K=2 | `112.648s` | `94.86 tok/s` | `27.06 tok/s` | `3048` | Still below baseline |
| K=3 | `96.563s` | `110.66 tok/s` | `31.19 tok/s` | `3012` | Slight gain |
| K=4 | `88.347s` | `120.95 tok/s` | `33.81 tok/s` | `2987` | Useful gain |
| K=5 | `82.134s` | `130.10 tok/s` | `35.81 tok/s` | `2941` | Useful gain |
| K=6 | `81.806s` | `130.63 tok/s` | `36.71 tok/s` | `3003` | Useful gain |
| K=7 | `81.041s` | `131.86 tok/s` | `37.68 tok/s` | `3054` | Numeric best in this sweep |
| K=8 | `81.223s` | `131.56 tok/s` | `36.81 tok/s` | `2990` | First decline; sweep stopped |

Interpretation: Gemma's optimal MTP point is workload-sensitive. The LongGen3
numeric best is K=7, improving decode-request throughput from `30.67` to
`37.68 tok/s`, about `+22.9%`. This is a useful gain, but smaller than Qwen's
MTP gain and not a universal default.

## Gemma PP4096/TG128 Peak Recheck

Under locked high GPU clocks, Gemma4 31B GPTQ FP16/default KV + assistant MTP5
also produced a strong synthetic PP4096/TG128 row:

| Route | Warm prefill | Warm decode | Notes |
| --- | ---: | ---: | --- |
| FP16/default KV + MTP5 | `1655.65 tok/s` | `99.64 tok/s` | Warm peak decode row |
| TQ4NC + MTP5 | `1660.13 tok/s` | `11.66 tok/s` | Starts, but speculative chunking is ineffective |

This makes Gemma FP16/default KV a credible short-context speed route, while
Gemma TQ4NC + MTP remains a compatibility path rather than a speed path.

## Provenance

Copied from private lab logs and the 2026-06-02
`qwopus_gemma_longgen_mtp_sweep` report.
