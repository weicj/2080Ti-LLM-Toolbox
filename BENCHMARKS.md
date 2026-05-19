# Benchmarks

All performance rows must keep prefill and decode separate. Aggregate tokens per
second are not enough for this repository.

Unless noted otherwise, rows are **single-request** serving measurements: one
active request at a time. Multi-concurrency batching numbers must be marked
separately because they are not directly comparable with llama.cpp single-slot
serving.

## Qwen3.6-27B-AWQ

Hardware for the vLLM rows: dual modified RTX 2080 Ti 22GB, NVLink, TP=2, one
TU102-300A card plus one TU102-300 card.

## Single-Request Microbenchmarks

| Engine | Config | Prompt / Generate | Prefill | Decode | E2E | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vLLM | MTP off | 4K / 128 | `1858 tok/s` | `45.44 tok/s` | `5.05s` | Sweep baseline |
| vLLM | MTP K=1 | 4K / 128 | `1814.1 tok/s` | `47.08 tok/s` | `4.996s` | Small decode gain |
| vLLM | MTP K=2 | 4K / 128 | `1845.9 tok/s` | `63.42 tok/s` | `4.237s` | Good acceptance |
| vLLM | MTP K=3 | 4K / 128 | `1843.7 tok/s` | `79.14 tok/s` | `3.839s` | Best current 4K result |
| vLLM | MTP K=4 | 4K / 128 | `1856.0 tok/s` | `72.03 tok/s` | `3.984s` | Fourth token acceptance regressed |
| vLLM | MTP off | 64K / 512 | `1326 tok/s` | `36.22 tok/s` | `62.41s` | Sweep baseline |
| vLLM | MTP K=1 | 64K / cap | `1301.2 tok/s` | `48.30 tok/s` | `57.798s` | EOS before strict 512 |
| vLLM | MTP K=2 | 64K / cap | `1295.4 tok/s` | `53.04 tok/s` | `57.115s` | EOS before strict 512 |
| vLLM | MTP K=3 | 64K / cap | `1294.3 tok/s` | `55.33 tok/s` | `56.768s` | Best current 64K result, EOS at 405 tokens |
| llama.cpp | GGUF single 2080 Ti | 4114 / 128 | `553.38 tok/s` | `23.74 tok/s` | `12.84s` | T1 backend `18101`, `cache_prompt=false` |
| llama.cpp | GGUF single 2080 Ti | 64022 / 512 | `383.12 tok/s` | `16.29 tok/s` | `198.63s` | T1 backend `18101`, `cache_prompt=false` |

## Sequential 60-Request Serving Run

The Ragent6 run is used here as a realistic sequential serving workload: 60
agent-style requests issued one after another against the same server. The model
score is model-specific and is not the point of this toolbox; serving throughput
is the hardware/runtime signal.

vLLM MTP K=3, Qwen3.6-27B-AWQ:

- requests: `60`, sequential, no concurrent batching
- wall: `167.39s`
- prompt throughput median/mean/max: `867.8/700.9/1306.4 tok/s`
- generation throughput median/mean/max: `31.8/35.2/71.6 tok/s`
- speculative accepted/drafted: `4501/5616 = 80.15%`
- quality sanity only: strict `43/60`, partial weighted `82.5/100`

Comparable earlier llama.cpp 27B runs on the same 60-request style workload:

- llama.cpp baseline: `471s` wall, PP mean `350.34 tok/s`, TG mean
  `21.15 tok/s`
- llama.cpp MTP n=2: `306s` wall, PP mean `297.01 tok/s`, TG mean
  `45.10 tok/s`, draft acceptance `80.42%`

SGLang is intentionally excluded from these comparison tables until it has a
valid prefill/decode benchmark and repeated-request run. Current SGLang status
is documented as compatibility bring-up in
[models/qwen3.6-27b-awq/sglang-smoke.md](models/qwen3.6-27b-awq/sglang-smoke.md).

## Other Models

| Model | Engine | Status | Prefill | Decode | Quality note |
| --- | --- | --- | ---: | ---: | --- |
| zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ | vLLM AWQ slow path | Not recommended for online serving | `1242.7 tok/s` 4K | `5.27 tok/s` 4K | Decent quality, unusable decode speed |
| feanors/Qwen3.6-35B-A3B Claude/Opus distilled AWQ | vLLM AWQ Marlin | Not recommended due model quality | `3235.3 tok/s` 4K | `98.4 tok/s` 4K | Fast but failed agent workload quality |
| cyankiwi/gemma-4-31B-it-AWQ-4bit | vLLM compressed-tensors | Blocked on SM75 shared memory | - | - | No completed prefill |

## Provenance

Numbers were copied from private lab logs dated 2026-05-17 through 2026-05-19.
Machine-local paths are intentionally not published in this repository.
