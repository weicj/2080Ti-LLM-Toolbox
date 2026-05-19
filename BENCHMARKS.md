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

### 4096 / 128

| Engine | Config | Prefill | Decode | E2E | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| vLLM | MTP off | `1858.0 tok/s` | `45.4 tok/s` | `5.1s` | Sweep baseline |
| vLLM | MTP K=1 | `1814.1 tok/s` | `47.1 tok/s` | `5.0s` | Small decode gain |
| vLLM | MTP K=2 | `1845.9 tok/s` | `63.4 tok/s` | `4.2s` | Good acceptance |
| vLLM | MTP K=3 | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` | Best current 4K result |
| vLLM | MTP K=4 | `1856.0 tok/s` | `72.0 tok/s` | `4.0s` | Fourth token acceptance regressed |
| llama.cpp | GGUF single 2080 Ti | `553.4 tok/s` | `23.7 tok/s` | `12.8s` | T1 backend `18101`, `cache_prompt=false` |

### 64K / 512

| Engine | Config | Prefill | Decode | E2E | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| vLLM | MTP off | `1326.0 tok/s` | `36.2 tok/s` | `62.4s` | Sweep baseline |
| vLLM | MTP K=1 | `1301.2 tok/s` | `48.3 tok/s` | `57.8s` | EOS before strict 512 |
| vLLM | MTP K=2 | `1295.4 tok/s` | `53.0 tok/s` | `57.1s` | EOS before strict 512 |
| vLLM | MTP K=3 | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` | Best current 64K result, EOS at 405 tokens |
| llama.cpp | GGUF single 2080 Ti | `383.1 tok/s` | `16.3 tok/s` | `198.6s` | T1 backend `18101`, `cache_prompt=false` |

## Sequential 60-Request Serving Run

The Ragent6 run is used here as a realistic sequential serving workload: 60
agent-style requests issued one after another against the same server. The model
score is model-specific and is not the point of this toolbox; serving throughput
is the hardware/runtime signal. Ragent6 lives at
[weicj/Ragent6](https://github.com/weicj/Ragent6).

vLLM MTP K=3, Qwen3.6-27B-AWQ:

- requests: `60`, sequential, no concurrent batching
- wall: `167.4s`
- prompt throughput median/mean/max: `867.8/700.9/1306.4 tok/s`
- generation throughput median/mean/max: `31.8/35.2/71.6 tok/s`
- speculative accepted/drafted: `4501/5616 = 80.2%`
- quality sanity only: strict `43/60`, partial weighted `82.5/100`

Comparable earlier llama.cpp 27B runs on the same 60-request style workload:

- llama.cpp baseline: `471.0s` wall, PP mean `350.3 tok/s`, TG mean
  `21.2 tok/s`
- llama.cpp MTP n=2: `306.0s` wall, PP mean `297.0 tok/s`, TG mean
  `45.1 tok/s`, draft acceptance `80.4%`

SGLang is intentionally excluded from these comparison tables until it has a
valid prefill/decode benchmark and repeated-request run. Current SGLang status
is documented as compatibility bring-up in
[models/qwen3.6-27b-awq/sglang-smoke.md](models/qwen3.6-27b-awq/sglang-smoke.md).

## Other Models

| Model | Engine | Status | Prefill | Decode | Quality note |
| --- | --- | --- | ---: | ---: | --- |
| zhiqing/Huihui-Qwen3.6-27B-abliterated-AWQ | vLLM AWQ slow path | Not recommended for online serving | `1242.7 tok/s` 4K | `5.3 tok/s` 4K | Decent quality, unusable decode speed |
| feanors/Qwen3.6-35B-A3B Claude/Opus distilled AWQ | vLLM AWQ Marlin | Not recommended due model quality | `3235.3 tok/s` 4K | `98.4 tok/s` 4K | Fast but failed agent workload quality |
| cyankiwi/gemma-4-31B-it-AWQ-4bit | vLLM compressed-tensors | Blocked on SM75 shared memory | - | - | No completed prefill |

## Provenance

Numbers were copied from private lab logs dated 2026-05-17 through 2026-05-19.
Machine-local paths are intentionally not published in this repository.
