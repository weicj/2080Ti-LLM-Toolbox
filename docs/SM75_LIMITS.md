# SM75 Limits That Matter

RTX 2080 Ti is Turing SM75. The common failure mode in modern LLM stacks is not
"CUDA unavailable"; it is that newer kernels quietly assume Ampere-or-newer
capabilities.

Observed constraints:

- Shared memory per block can be too small for newer Triton attention kernels.
  The Gemma4 31B AWQ vLLM attempt required 96KB while SM75 exposed 64KB.
- Some prebuilt fused kernels do not ship SM75 images.
- BF16-oriented fast paths usually need fallback or FP16 routing.
- Python-level tensor allocation inside hot CUDA paths can trip unsupported
  kernels when the upstream code assumes newer GPU support.
- Long-context serving may be possible, but the viable path can differ sharply
  between prefill and decode.

Design rule for this repo: record which operation fails, which backend selected
it, and whether the failure is a kernel image issue, shared-memory issue,
quantization layout issue, or quality issue.

