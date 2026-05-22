# Huihui Qwen3.6-27B Abliterated AWQ MTP

Model: `zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP`

Status: first-batch candidate, pending Marlin gate.

Route target:

- vLLM TP=2
- `awq_marlin`
- MTP K=3
- same benchmark shape as baseline `QuantTrio/Qwen3.6-27B-AWQ`

## Why This Candidate

- Same 27B family and AWQ direction as current working route.
- Includes MTP-oriented variant, so it is a direct practical candidate for the
  existing serving shape.

## Known Risk

- Previous non-MTP Huihui AWQ runs on this rig fell back to slow AWQ path and
  did not sustain Marlin fast path.
- If startup logs do not show Marlin kernel path, this candidate should be
  rejected immediately for performance target use.

## Gate Checklist

- [ ] Marlin gate passes (`MarlinLinearKernel` or equivalent)
- [ ] PP4096/TG128 decode no worse than baseline stable line
- [ ] PP64K/TG512 decode no meaningful regression
- [ ] Ragent6 10-case quality and stability pass
- [ ] Optional full60 pass for promotion

See shared gate template:
[../qwen3.6-27b-marlin-candidates-template.md](../qwen3.6-27b-marlin-candidates-template.md)
