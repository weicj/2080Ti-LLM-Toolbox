# Hardware

The first benchmark target is a dual-card Turing system:

- 2x modified RTX 2080 Ti 22GB
- NVLink between the two cards
- one TU102-300A GPU
- one TU102-300 GPU
- tensor parallel size 2 for the main vLLM results

This matters because the headline vLLM route is not expected to fit or behave
the same way on a stock 11GB RTX 2080 Ti. Single-card 11GB users should treat
the current results as SM75 compatibility evidence, not as a direct capacity
promise.

Benchmark policy:

- Prefer single-request serving measurements.
- Mark multi-concurrency or batching tests explicitly.
- Always report prefill and decode separately.
- Keep NVLink/topology notes attached to TP=2 results.

