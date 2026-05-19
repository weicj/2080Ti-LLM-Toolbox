# vLLM on RTX 2080 Ti / SM75

Current best tested path:

- vLLM 0.21
- torch 2.11 cu130
- flashinfer 0.6.8
- dual RTX 2080 Ti, TP=2
- Qwen3.6-27B-AWQ
- AWQ Marlin
- FlashInfer full attention
- FlashQLA SM70/SM75 legacy GDN prefill
- MTP K=3

The useful local patch was to let vLLM select our FlashQLA SM70/SM75 legacy
backend for GDN prefill via
`--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'`.
The FlashQLA legacy output buffer also had to match `[B,T,Hv,D]` for
`linear_num_key_heads=16`, `linear_num_value_heads=48`.

Known caveat: the stack still uses Triton in several places, including slot
mapping, causal conv, fused post-conv, and decode GDN update. This is not a
"no Triton" route.

See:

- [models/qwen3.6-27b-awq/vllm-mtp-k3.md](../../models/qwen3.6-27b-awq/vllm-mtp-k3.md)
- [BENCHMARKS.md](../../BENCHMARKS.md)
