# Heretic Qwen3.6-27B GPTQ Int4 Native MTP Preserved

Model: `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`

Status: first-batch candidate, pending GPTQ-Marlin gate.

Route target:

- vLLM TP=2
- `gptq_marlin`
- MTP K=3
- same benchmark shape as baseline `QuantTrio/Qwen3.6-27B-AWQ`

## Why This Candidate

- Explicit GPTQ Int4 candidate with MTP-preserved variant.
- Uncensored/community objective matches current exploration target.

## Known Risk

- GPTQ models can run but miss `gptq_marlin` and fall back to slower kernels.
- If startup logs do not show Marlin-compatible GPTQ fast path, reject for
  performance-critical route.

## Gate Checklist

- [ ] GPTQ-Marlin gate passes (`gptq_marlin`/Marlin kernel evidence)
- [ ] PP4096/TG128 decode no worse than baseline stable line
- [ ] PP64K/TG512 decode no meaningful regression
- [ ] Ragent6 10-case quality and stability pass
- [ ] Optional full60 pass for promotion

See shared gate template:
[../qwen3.6-27b-marlin-candidates-template.md](../qwen3.6-27b-marlin-candidates-template.md)
