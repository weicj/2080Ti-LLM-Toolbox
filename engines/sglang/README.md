# SGLang on RTX 2080 Ti / SM75

Status: experimental.

Two SGLang paths were tested:

## Qwen3.6-27B GGUF

Clean SGLang PR19585 environment:

- torch `2.9.1/cu128`
- flashinfer `0.6.3`
- sgl-kernel `0.3.21`
- transformers `4.57.1`

Result:

- TP=2 failed on dense GGUF tensor slicing:
  `start (6144)+length(6144)>dimension(6144)`.
- TP=1 loaded weights (`mem usage=14.46 GB`, KV `26258` tokens), then failed in
  `sgl_kernel.rmsnorm` with `no kernel image is available`.

Conclusion: GGUF was not serviceable on SM75/NVLink in that test.

## Qwen3.6-27B-AWQ

Short `/generate` smoke eventually reached HTTP 200 on temporary port `19182`
using SGLang + FlashInfer + FlashQLA legacy + SM75 fallbacks:

- `prompt_tokens=5`
- `completion_tokens=2`
- `e2e_latency=3.67s`
- output: `ieee!`

That proves execution only. It does not validate quality or performance.

Workaround categories recorded:

- `ReqToTokenPool.write` Triton scatter
- vocab mask Triton
- LM head Triton linear
- mamba prefix cat
- decode allocation avoiding `clone`/`to`
- schedule batch add
- `clamp_position`
- FlashInfer decode cumsum
- hybrid decode arange
- `get_mamba_indices` Triton gather
- `query_start_loc` initialization
- embedding and mask row zero
- FLA contiguous bypass
- FLA buffer/preallocation work

TODO: export patch queue from the experiment venv instead of keeping this as a
memory-derived list.

