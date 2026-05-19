# Architecture

Club2080Ti is organized as an integration toolbox:

- Engine notes live under `engines/`.
- Model-specific recipes live under `models/`.
- Reproducibility locks live under `manifests/`.
- Curated benchmark summaries live under `reports/summaries/`.
- Raw logs are referenced, not committed by default.

Upstream projects remain separate. This repository should not vendor vLLM,
SGLang, FlashInfer, FlashQLA, or FLA. It should instead pin versions, store patch
queues, and provide scripts that rebuild the tested stack.

