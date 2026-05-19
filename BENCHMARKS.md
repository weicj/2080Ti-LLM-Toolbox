# Benchmarks

All performance rows must keep prefill and decode separate. Aggregate tokens per
second are not enough for this repository.

Unless noted otherwise, rows are **single-request** serving measurements: one
active request at a time. Multi-concurrency batching numbers must be marked
separately because they are not directly comparable with llama.cpp single-slot
serving.

## Qwen3.6 27B Artifacts

Hardware for the vLLM rows: dual modified RTX 2080 Ti 22GB, NVLink, TP=2, one
TU102-300A card plus one TU102-300 card. llama.cpp baseline rows use a single
modified RTX 2080 Ti 22GB unless noted otherwise.

Only Qwen3.6 27B-family artifacts are listed in the main benchmark tables.
Derivative model lines and non-27B routes are excluded from this scorecard.

| Artifact | Engine Route | Size / Quantization | MTP / Draft Status |
| --- | --- | --- | --- |
| `QuantTrio-Qwen3.6-27B-AWQ` | vLLM / SGLang | 21.9 GB safetensors, AWQ 4-bit, group size 128, zero point enabled | MTP tensors present; vLLM validated with AWQ Marlin K=1..4 |
| `unsloth/Qwen3.6-27B-GGUF` `Qwen3.6-27B-Q4_K_M.gguf` | llama.cpp upstream-original baseline | 16.8 GB file, GGUF `Q4_K_M` | No integrated MTP head |
| RDson `Qwen3.6-27B-MTP-Q4_K_M.gguf` | llama.cpp integrated MTP | 16.5 GB file, GGUF `Q4_K_M` with MTP tensors | Used for PR22673 MTP n=2/n=3 rows |

## Single-Request Microbenchmarks

### PP4096 / TG128

| Engine | Route | Model | Prefill | Decode | E2E | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | MTP off | Qwen3.6-27B-AWQ | `1858.0 tok/s` | `45.4 tok/s` | `5.1s` | Sweep baseline |
| vLLM | MTP K=1 | Qwen3.6-27B-AWQ | `1814.1 tok/s` | `47.1 tok/s` | `5.0s` | Small decode gain |
| vLLM | MTP K=2 | Qwen3.6-27B-AWQ | `1845.9 tok/s` | `63.4 tok/s` | `4.2s` | Good acceptance |
| vLLM | MTP K=3 | Qwen3.6-27B-AWQ | `1843.7 tok/s` | `79.1 tok/s` | `3.8s` | Best current 4K result |
| vLLM | MTP K=4 | Qwen3.6-27B-AWQ | `1856.0 tok/s` | `72.0 tok/s` | `4.0s` | Fourth token acceptance regressed |
| llama.cpp | upstream-original GGUF baseline | Qwen3.6-27B-Q4_K_M | `553.4 tok/s` | `23.7 tok/s` | `12.8s` | Single 2080 Ti, `cache_prompt=false` |

### PP64K / TG512

| Engine | Route | Model | Prefill | Decode | E2E | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | MTP off | Qwen3.6-27B-AWQ | `1326.0 tok/s` | `36.2 tok/s` | `62.4s` | Sweep baseline |
| vLLM | MTP K=1 | Qwen3.6-27B-AWQ | `1301.2 tok/s` | `48.3 tok/s` | `57.8s` | EOS before strict 512 |
| vLLM | MTP K=2 | Qwen3.6-27B-AWQ | `1295.4 tok/s` | `53.0 tok/s` | `57.1s` | EOS before strict 512 |
| vLLM | MTP K=3 | Qwen3.6-27B-AWQ | `1294.3 tok/s` | `55.3 tok/s` | `56.8s` | Best current 64K result, EOS at 405 tokens |
| llama.cpp | upstream-original GGUF baseline | Qwen3.6-27B-Q4_K_M | `383.1 tok/s` | `16.3 tok/s` | `198.6s` | Single 2080 Ti, `cache_prompt=false` |

### PP16K / TG4096

This table uses the RDson integrated-MTP GGUF artifact, not the official
baseline GGUF. It is kept in a separate workload table because prompt/generation
length differs from the PP4096/TG128 and PP64K/TG512 rows.

| Engine | Route | Model | Prefill | Decode | E2E | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| llama.cpp | baseline, same MTP GGUF artifact | RDson Qwen3.6-27B-MTP-Q4_K_M | `609.4 tok/s` | `18.5 tok/s` | `248.9s` | MTP disabled |
| llama.cpp | integrated MTP n=2 | RDson Qwen3.6-27B-MTP-Q4_K_M | `501.3 tok/s` | `28.4 tok/s` | `177.0s` | Draft acceptance `68.0%` |
| llama.cpp | integrated MTP n=3 | RDson Qwen3.6-27B-MTP-Q4_K_M | `496.8 tok/s` | `27.7 tok/s` | `181.2s` | Draft acceptance `60.7%` |

## Sequential 60-Request Serving Run

The Ragent6 run is used here as a realistic sequential serving workload: 60
agent-style requests issued one after another against the same server. The model
score is model-specific and is not the point of this toolbox; serving throughput
is the hardware/runtime signal. Ragent6 lives at
[weicj/Ragent6](https://github.com/weicj/Ragent6).

| Engine | Route | Model | Prefill | Decode | E2E | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| vLLM | TP=2, MTP K=2 | Qwen3.6-27B-AWQ | `631.9 tok/s` | `31.3 tok/s` | `190.6s` | Accepted/drafted `85.9%`; slower than K=3 |
| vLLM | TP=2, MTP K=3 | Qwen3.6-27B-AWQ | `700.9 tok/s` | `35.2 tok/s` | `167.4s` | Ragent6 60-case run |
| llama.cpp | baseline, same MTP GGUF artifact | RDson Qwen3.6-27B-MTP-Q4_K_M | `350.3 tok/s` | `21.2 tok/s` | `471.0s` | MTP disabled |
| llama.cpp | integrated MTP n=2 | RDson Qwen3.6-27B-MTP-Q4_K_M | `297.0 tok/s` | `45.1 tok/s` | `306.0s` | Draft acceptance `80.4%` |

SGLang is intentionally excluded from these comparison tables until it has a
valid prefill/decode benchmark and repeated-request run. Current SGLang status
is documented as compatibility bring-up in
[models/qwen3.6-27b-awq/sglang-smoke.md](models/qwen3.6-27b-awq/sglang-smoke.md).

## Provenance

Numbers were copied from private lab logs dated 2026-05-17 through 2026-05-19.
Machine-local paths are intentionally not published in this repository.
