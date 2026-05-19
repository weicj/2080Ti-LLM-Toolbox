# Qwen3.6-27B: llama.cpp Baseline

Status: baseline comparison.

Known result on one RTX 2080 Ti 22GB from the same lab rig,
`cache_prompt=false`:

| Prompt / Generate | Prefill | Decode | E2E |
| ---: | ---: | ---: | ---: |
| 4114 / 128 | `553.38 tok/s` | `23.74 tok/s` | `12.84s` |
| 64022 / 512 | `383.12 tok/s` | `16.29 tok/s` | `198.63s` |

Interpretation:

- llama.cpp decode is strong for a single-card GGUF baseline.
- vLLM with AWQ + FlashQLA + MTP K=3 is much stronger for the current 64K
  long-prompt route.
