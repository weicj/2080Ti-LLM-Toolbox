# Qwen3.6-27B-AWQ: SGLang Smoke

Status: experimental, not production-ready.

The first successful execution smoke used temporary port `19182` and returned:

```text
HTTP 200
prompt_tokens = 5
completion_tokens = 2
e2e_latency = 3.67s
output = ieee!
```

The output was wrong for the test prompt. This only proves that the stack can
reach a generate response after many SM75 fallbacks.

Stack:

- SGLang
- FlashInfer
- FlashQLA SM70/SM75 legacy backend from
  [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75)
- multiple SM75 Triton fallbacks
- dual RTX 2080 Ti

Known patch categories are tracked in
[engines/sglang/README.md](../../engines/sglang/README.md).

No valid prefill/decode benchmark exists yet for this route.
