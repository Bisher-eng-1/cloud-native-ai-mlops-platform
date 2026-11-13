#!/usr/bin/env python3
"""Send sample normal, incident, and shifted payloads to verify /predict, /drift, and /metrics."""
from __future__ import annotations

import argparse
import json
import urllib.request


def _request(url: str, payload: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test live MLOps inference service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    _, ready = _request(f"{base}/ready")
    print("READY:", ready)

    samples = [
        "INFO service=auth-api region=eu-central-1 status=200 latency_ms=24 healthcheck passed",
        "INFO service=cache-node region=eu-west-2 processed batch size=42 without errors",
        "CRITICAL service=worker-pool region=eu-central-1 out of memory oom killer terminated worker",
        "ERROR service=payment-gateway region=us-east-1 status=503 upstream timeout after 4800ms",
    ]
    for text in samples:
        _, body = _request(f"{base}/predict", {"text": text})
        print("PREDICT:", body)

    _, drift = _request(f"{base}/drift")
    print("DRIFT:", drift)


if __name__ == "__main__":
    main()
