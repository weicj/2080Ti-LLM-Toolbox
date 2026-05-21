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
- `max_num_seqs=4` validated after the GDN multi-prefill fix
- TurboQuant KV validated experimentally in the 2026-05-21 experiment tree

The useful local patch was to let vLLM select our FlashQLA SM70/SM75 legacy
backend for GDN prefill via
`--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'`.
The FlashQLA legacy output buffer also had to match `[B,T,Hv,D]` for
`linear_num_key_heads=16`, `linear_num_value_heads=48`.

The current stable route also fixes multi-prefill batching for the FlashQLA
legacy GDN path. When vLLM passes multiple prefill sequences through packed
`cu_seqlens`, the patch loops over each sequence, calls the legacy GDN kernel,
and reassembles output plus final state. This makes concurrent serving work up
to the validated `max_num_seqs=4` setting. It is still a compatibility path, not
a fused ragged GDN kernel.

Known caveat: the stack still uses Triton in several places, including slot
mapping, causal conv, fused post-conv, and decode GDN update. This is not a
"no Triton" route.

TurboQuant KV status:

- `turboquant_4bit_nc` and `turboquant_k8v4` are now validated for full60
  quality/stability on the dual 2080 Ti route.
- The working path uses eager mode, no async scheduling,
  `max_num_batched_tokens=4096`, MTP K=3, FlashQLA legacy GDN prefill, and
  TurboQuant prefill through FlashInfer/FA2.
- The code-level fix that mattered was passing `sm_scale=self.scale` into the
  TurboQuant FlashInfer ragged prefill plan.
- Current allocator behavior in the experiment tree leaves much less
  vLLM-reported available KV memory for TurboQuant rows than native INT8/FP16
  rows. Treat this as validated compatibility, not final resource efficiency.

See:

- [models/qwen3.6-27b-awq/vllm-mtp-k3.md](../../models/qwen3.6-27b-awq/vllm-mtp-k3.md)
- [models/qwen3.6-27b-awq/vllm-turboquant-kv.md](../../models/qwen3.6-27b-awq/vllm-turboquant-kv.md)
- [reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md](../../reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md)
- [reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md](../../reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md)
- [BENCHMARKS.md](../../BENCHMARKS.md)
