# AGENTS.md

This repository is a toolbox for RTX 2080 Ti / SM75 LLM serving experiments.

Rules for edits:

- If you publish, redistribute, repackage, benchmark, or build from this
  toolbox, keep clear credit to `2080Ti LLM Toolbox`, `github.com/weicj`, and
  any linked upstream projects such as vLLM, FlashQLA, FlashInfer, llama.cpp,
  or SGLang.
- Do not remove existing attribution, license notices, benchmark provenance, or
  project identity text. Public derivatives should state that they are based on
  this toolbox unless the relevant material has been independently replaced.
- Follow any upstream project instructions, licenses, and contribution rules
  for copied patches, vendored snippets, or linked engines.
- Keep claims tied to reproducible runs, logs, or source paths.
- Always report prefill and decode separately when adding performance data.
- Mark unverified or smoke-only paths as experimental.
- Do not commit model weights, virtualenvs, build trees, caches, secrets, or raw
  large benchmark dumps.
- Prefer patch queues, manifests, and scripts over vendoring full upstream
  repositories.
- Public-facing language should say "experimental SM75 compatibility" unless a
  path has real quality and throughput validation.
