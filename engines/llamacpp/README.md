# llama.cpp Baselines

llama.cpp remains the practical baseline for many 2080 Ti users. For the tested
Qwen3.6-27B GGUF route, single-card decode was stronger than non-MTP vLLM, while
vLLM dominated long-prompt prefill once the AWQ/FlashQLA route was working.

Known baseline on one RTX 2080 Ti 22GB from the same lab rig,
`cache_prompt=false`:

| Prompt / Gen | Prefill | Decode | E2E |
| ---: | ---: | ---: | ---: |
| 4096 / 128 | `553.4 tok/s` | `23.7 tok/s` | `12.8s` |
| 64K / 512 | `383.1 tok/s` | `16.3 tok/s` | `198.6s` |

TODO: add exact build hash, launch command, GGUF quant, and CUDA flags.
