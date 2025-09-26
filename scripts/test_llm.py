#!/usr/bin/env python
"""Quick utility for sanity-checking an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import httpx

DEFAULT_BASE = "http://10.1.1.35:38411/v1"


def _request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    stream: bool,
    timeout: float,
) -> httpx.Response:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Explicitly set the stream flag so gateways don't default to SSE.
    payload["stream"] = bool(stream)

    client = httpx.Client(timeout=timeout)
    try:
        response = client.post(endpoint, headers=headers, json=payload)
    finally:
        client.close()
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to send to the chat completion API (positional form).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRAPHRAG_API_BASE", DEFAULT_BASE),
        help="OpenAI-compatible base URL (defaults to GRAPHRAG_API_BASE or the local config value).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GRAPHRAG_API_KEY"),
        required=os.environ.get("GRAPHRAG_API_KEY") is None,
        help="API key for the endpoint (defaults to GRAPHRAG_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4o"),
        help="Model identifier to request.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Include 'stream': true and dump the raw server response to stdout.",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_override",
        help="Prompt to send to the API (flag form). Overrides positional argument if provided.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds.",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set GRAPHRAG_API_KEY)")

    prompt = args.prompt_override if args.prompt_override is not None else args.prompt
    if prompt is None:
        prompt = "Say hello to GraphRAG"

    try:
        response = _request(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            stream=args.stream,
            timeout=args.timeout,
        )
    except Exception as exc:  # pragma: no cover - interactive utility
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"HTTP {response.status_code} {response.reason_phrase}")
    print("-- response headers --")
    for key, value in response.headers.items():
        print(f"{key}: {value}")

    print("\n-- response body --")
    body: Optional[str] = None
    try:
        body = response.text
    except Exception as exc:  # pragma: no cover - defensive
        print(f"<failed to read body: {exc}>")

    if body is None:
        print("<no body>")
    else:
        if not args.stream:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                print(body)
            else:
                print(json.dumps(parsed, indent=2))
        else:
            print(body)

    if response.is_error:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
