# Qwen3.6-27B-AWQ vLLM MTP K=3 Summary

Hardware: dual modified RTX 2080 Ti 22GB, NVLink, TP=2; one TU102-300A GPU and
one TU102-300 GPU.

Stack: vLLM 0.21, torch 2.11 cu130, FlashInfer attention, FlashQLA SM70/SM75
legacy GDN prefill, AWQ Marlin, MTP K=3.

Performance:

- 4K/tg128: `1843.7 tok/s` prefill, `79.14 tok/s` decode, `3.839s` e2e.
- 64K/tg512 cap: `1294.3 tok/s` prefill, `55.33 tok/s` decode, `56.768s` e2e.

Sequential 60-request serving run:

- workload: 60 real agent-style requests, one active request at a time.
- wall: `167.39s`.
- prompt throughput median/mean/max: `867.8/700.9/1306.4 tok/s`.
- generation throughput median/mean/max: `31.8/35.2/71.6 tok/s`.
- quality sanity: strict `43/60`, partial weighted `82.5/100`.

Interpretation:

This is the first route in this toolbox that is both fast enough and quality
validated enough to be called a recommended path.
