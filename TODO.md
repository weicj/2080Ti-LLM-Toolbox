# TODO

## Before Public Release

- Add exact upstream commit hashes for vLLM, SGLang, FlashInfer, FlashQLA, FLA,
  PyTorch, and CUDA wheels used in the miniclaw experiments.
- Export patch queues from the SGLang and vLLM site-package experiments instead
  of relying on memory notes.
- Add a strict `ignore_eos` 64K/tg512 rerun for vLLM MTP K=3.
- Add a reproducible setup script that clones pinned upstream repositories and
  applies patch queues.
- Decide public license.
- Rewrite any machine-specific paths into examples while preserving the original
  source pointers in private notes.

## Nice To Have

- Add single-2080Ti recipes.
- Add 2080 Ti NVLink topology checklist.
- Add thermal and fan-curve recommendations.
- Add a matrix for CUDA 12.8, CUDA 13.0, and driver compatibility.

