# FlashQLA SM70/SM75

FlashQLA SM70/SM75 is our separate kernel project and a first-class foundation
for this toolbox. It is not just a generic external dependency.

Current fork:

```text
repo: https://github.com/weicj/FlashQLA-SM70-SM75
branch: sm70-sm75-gdn-forward
known public commit: 3ab27d77d8ca
```

Why it matters:

- Qwen3.5/3.6 GDN prefill needs a kernel path that can run on SM75.
- vLLM and SGLang can both benefit from a reusable legacy GDN backend.
- Keeping FlashQLA separate makes upstream PRs and kernel sanity tests easier.

Known local follow-up:

- The FlashQLA legacy `gdn_forward.cu` output buffer fix for
  `[B,T,Hv,D]` was important when `linear_num_key_heads=16` and
  `linear_num_value_heads=48`.
- The local FlashQLA working tree may contain newer unpushed experiment edits;
  do not treat this repo's commit pointer as the final patch queue.
