# vLLM on RTX 2080 Ti / SM75

Packaged runtime:
[weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
is the maintained 2080 Ti Definitive vLLM runtime for the SM75 patches,
launcher, profiles, and documentation referenced by this toolbox.

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
- A `max_model_len=262144` startup/cache probe succeeded for FP16, native INT8,
  `turboquant_k8v4`, and `turboquant_4bit_nc`. The largest measured row was
  `turboquant_4bit_nc` with `735,084` vLLM-reported KV tokens at about
  `20.6 GiB` total used VRAM per 2080 Ti.
- Treat the 262K probe as startup/cache/VRAM evidence, not full long-prompt
  throughput validation. The VRAM number is total device memory, not KV-only
  footprint.

Gemma4/Gemma31B note:

- The Qwen27 TurboQuant performance route was not lost. The later Gemma4/Gemma31B
  `tq4nc` prefill regression was a separate `head_dim=512` final full-attention
  issue.
- The fixed Gemma4 no-MTP route pairs
  `GEMMA4_TQ4NC_HEAD512_PREFILL_SDPA_FALLBACK=1` with
  `GEMMA4_SM75_SDPA_PREFILL512=1`. PP4096/TG128 recovered from old tq4nc median
  `1248.70 prefill / 32.98 decode tok/s` to `1646.56 / 33.23`, or
  `97.2% / 98.4%` of the same-build non-TQ baseline.
- This is an SDPA/Torch compatibility fallback, not yet a true SM75
  FlashInfer/FA2 head512 kernel.

See:

- [engines/vllm/recipes/qwen36-27b-awq-best-sm75.md](recipes/qwen36-27b-awq-best-sm75.md)
- [engines/vllm/patches](patches/README.md)
- [models/qwen3.6-27b-awq/vllm-mtp-k3.md](../../models/qwen3.6-27b-awq/vllm-mtp-k3.md)
- [models/qwen3.6-27b-awq/vllm-turboquant-kv.md](../../models/qwen3.6-27b-awq/vllm-turboquant-kv.md)
- [reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md](../../reports/summaries/qwen36-27b-awq-vllm-gdn-concurrency.md)
- [reports/summaries/qwen36-27b-awq-vllm-ctx262k-kv-vram.md](../../reports/summaries/qwen36-27b-awq-vllm-ctx262k-kv-vram.md)
- [reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md](../../reports/summaries/qwen36-27b-awq-vllm-turboquant-kv.md)
- [reports/summaries/gemma4-31b-gptq-vllm-tq4nc-prefill-fix.md](../../reports/summaries/gemma4-31b-gptq-vllm-tq4nc-prefill-fix.md)
- [BENCHMARKS.md](../../BENCHMARKS.md)
