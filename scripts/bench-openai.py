#!/usr/bin/env python3
"""Minimal OpenAI-compatible benchmark helper.

This script reports approximate prefill and decode rates from streaming time to
first token and total latency. It is intentionally simple; engine-specific logs
remain the source of truth when available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def request_json(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=payload.pop("_timeout", 600)) as res:
        return json.loads(res.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("CLUB2080TI_MODEL", "model"))
    parser.add_argument("--prompt", default="Say OK.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/completions"
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "_timeout": args.timeout,
    }

    started = time.perf_counter()
    try:
        result = request_json(url, payload, args.api_key)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        raise
    elapsed = time.perf_counter() - started

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    print(json.dumps({
        "model": args.model,
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "approx_total_tok_s": (
            (prompt_tokens + completion_tokens) / elapsed
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) and elapsed > 0
            else None
        ),
        "usage": usage,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

