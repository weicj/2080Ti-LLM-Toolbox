# Engines

This directory tracks engine-specific SM75 notes. The repository is organized
around tested recipes, not around a single preferred runtime.

- `vllm`: best current route for Qwen3.6-27B-AWQ on dual RTX 2080 Ti.
- `sglang`: experimental compatibility work; smoke-only for now.
- `llamacpp`: baseline and production comparison points.
- `flashqla`: kernel dependency used by both vLLM and SGLang experiments.

