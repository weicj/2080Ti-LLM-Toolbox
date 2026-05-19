# AGENTS.md

This repository is a toolbox for RTX 2080 Ti / SM75 LLM serving experiments.

Rules for edits:

- Keep claims tied to reproducible runs, logs, or source paths.
- Always report prefill and decode separately when adding performance data.
- Mark unverified or smoke-only paths as experimental.
- Do not commit model weights, virtualenvs, build trees, caches, secrets, or raw
  large benchmark dumps.
- Prefer patch queues, manifests, and scripts over vendoring full upstream
  repositories.
- Public-facing language should say "experimental SM75 compatibility" unless a
  path has real quality and throughput validation.

