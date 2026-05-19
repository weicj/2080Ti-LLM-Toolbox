# Qwen3.6-27B: llama.cpp Baseline

Status: baseline comparison.

Known result on one RTX 2080 Ti 22GB from the same lab rig,
`cache_prompt=false`:

| Prompt / Gen | Prefill | Decode | E2E |
| ---: | ---: | ---: | ---: |
| 4096 / 128 | `553.4 tok/s` | `23.7 tok/s` | `12.8s` |
| 64K / 512 | `383.1 tok/s` | `16.3 tok/s` | `198.6s` |

Interpretation:

- llama.cpp decode is strong for a single-card GGUF baseline.
- vLLM with AWQ + FlashQLA + MTP K=3 is much stronger for the current 64K
  long-prompt route.
