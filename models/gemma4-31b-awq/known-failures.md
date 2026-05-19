# Gemma4 31B AWQ-4bit Known Failure

Model: `cyankiwi/gemma-4-31B-it-AWQ-4bit`

Status: blocked on vLLM + SM75.

The model could load to `/health` ready under vLLM 0.21 and torch 2.11 cu130 on
dual RTX 2080 Ti, but failed on the first `pp4096/tg128` prefill.

Root cause:

```text
TRITON_ATTN required shared memory: 98304 bytes
SM75 hardware limit: 65536 bytes
```

Other attempts:

- `--attention-backend flashinfer` failed because partial multimodal token full
  attention was not supported.
- `--attention-backend torch_sdpa` failed because the backend was not registered.
- Reducing `max_num_batched_tokens` to `3072` still failed with the same shared
  memory issue.

No valid throughput or Ragent6 score exists for this route.

