# llama.cpp Baselines

llama.cpp remains the practical baseline for many 2080 Ti users. For the tested
Qwen3.6-27B GGUF route, single-card decode was stronger than non-MTP vLLM, while
vLLM dominated long-prompt prefill once the AWQ/FlashQLA route was working.

Known baseline on one RTX 2080 Ti 22GB from the same lab rig,
`cache_prompt=false`:

| Prompt / Generate | Prefill | Decode | E2E |
| ---: | ---: | ---: | ---: |
| 4114 / 128 | `553.38 tok/s` | `23.74 tok/s` | `12.84s` |
| 64022 / 512 | `383.12 tok/s` | `16.29 tok/s` | `198.63s` |

TODO: add exact build hash, launch command, GGUF quant, and CUDA flags.
