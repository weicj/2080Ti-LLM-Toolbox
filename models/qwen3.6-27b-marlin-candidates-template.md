# Qwen3.6-27B Marlin Candidate Gate

Purpose: evaluate community 27B candidates under the same vLLM dual-2080Ti
route, with hard acceptance gates focused on Marlin fast path and no performance
regression.

Scope: AWQ-Marlin or GPTQ-Marlin candidates only.

## Baseline

Reference route:

- model: `QuantTrio/Qwen3.6-27B-AWQ`
- stack: vLLM 0.21, torch 2.11 cu130, TP=2, MTP K=3
- quant path: `awq_marlin`

Reference numbers:

- PP4096/TG128 stable decode line: `79.1 tok/s`
- PP4096/TG128 peak decode line: `101.3 tok/s` median, `101.5 tok/s` max
- PP64K/TG512 decode line: `55.3 tok/s`
- Ragent6 full60 wall: `167.4s`

## Hard Gates

Candidate must pass all gates below.

1. Marlin gate
   Log must show Marlin kernel path (`MarlinLinearKernel` or equivalent Marlin
   backend). If fallback to slow `awq`/`gptq` generic kernel appears, fail.
2. 4K gate
   PP4096/TG128 decode must not regress below baseline stable line.
3. 64K gate
   PP64K/TG512 decode must not show meaningful regression against baseline.
4. Quality gate
   Run Ragent6 10-case first. If quality is clearly degraded or unstable, fail.
5. Stability gate
   No traceback, HTTP 500, OOM, or engine crash in baseline workload.

## Test Sequence

1. Launch candidate with fixed baseline settings and target quantization mode.
2. Collect startup log and verify Marlin gate.
   Use:
   `scripts/check-vllm-marlin-log.sh /path/to/server.log awq`
   or
   `scripts/check-vllm-marlin-log.sh /path/to/server.log gptq`
3. Run PP4096/TG128 and PP64K/TG512.
4. Run Ragent6 10-case.
5. Only if all pass, run Ragent6 full60.

## Candidate Sheet

Copy this section per model.

```text
candidate:
hf_id:
quant_type: awq / gptq
target_marlin_mode: awq_marlin / gptq_marlin

marlin_gate: pass/fail
evidence_log:

pp4096_tg128_prefill:
pp4096_tg128_decode:
pp64k_tg512_prefill:
pp64k_tg512_decode:

ragent6_10_strict:
ragent6_10_weighted:
stability_notes:

decision: promote / hold / reject
reason:
```
