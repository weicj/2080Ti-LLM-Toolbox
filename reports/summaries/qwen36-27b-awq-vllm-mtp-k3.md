# Qwen3.6-27B-AWQ vLLM MTP K=3 Summary

Hardware: miniclaw dual RTX 2080 Ti, TP=2.

Stack: vLLM 0.21, torch 2.11 cu130, FlashInfer attention, FlashQLA legacy GDN
prefill, AWQ Marlin, MTP K=3.

Performance:

- 4K/tg128: `1843.7 tok/s` prefill, `79.14 tok/s` decode, `3.839s` e2e.
- 64K/tg512 cap: `1294.3 tok/s` prefill, `55.33 tok/s` decode, `56.768s` e2e.

Quality:

- Ragent6 0.2.2 zh-CN strict `43/60`.
- partial weighted `82.5/100`.
- wall `167.39s`.

Interpretation:

This is the first route in this toolbox that is both fast enough and quality
validated enough to be called a recommended path.

