# Models

Model notes are grouped by exact model family and quantization. Each model page
should state:

- engine compatibility
- quantization format
- whether the fast path is available
- prefill and decode numbers, separately
- quality score if available
- known failure mode if rejected

Current main model:

- [qwen3.6-27b-awq](qwen3.6-27b-awq/README.md)
- [qwen3.6-27b-marlin-candidates-template](qwen3.6-27b-marlin-candidates-template.md)

Current first-batch Marlin candidates:

- [huihui-qwen3.6-27b-awq-mtp](huihui-qwen3.6-27b-awq-mtp/README.md)
- [heretic-qwen3.6-27b-gptq-int4-mtp](heretic-qwen3.6-27b-gptq-int4-mtp/README.md)

Known rejected or not-yet-useful models:

- [huihui-qwen3.6-27b-awq](huihui-qwen3.6-27b-awq/README.md)
- [qwen3.6-35b-a3b-awq](qwen3.6-35b-a3b-awq/README.md)
- [gemma4-31b-awq](gemma4-31b-awq/known-failures.md)
