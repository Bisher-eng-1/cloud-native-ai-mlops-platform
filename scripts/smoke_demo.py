#!/usr/bin/env python3
"""Send sample normal, incident, and shifted payloads to verify /ready, /predict, /history, /drift, and /metrics."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def _request(url: str, payload: dict[str, Any] | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8")


def run_smoke(base_url: str, expect_backend: str | None = None) -> dict[str, Any]:
    base = base_url.rstrip("/")
    _, ready_raw = _request(f"{base}/ready")
    ready = json.loads(ready_raw)
    print("READY:", ready_raw)
    if ready.get("status") != "ready":
        raise RuntimeError(f"Service not ready: {ready}")
    if expect_backend and ready.get("storage_backend") != expect_backend:
        raise RuntimeError(
            f"Expected storage_backend={expect_backend!r}, got {ready.get('storage_backend')!r}"
        )

    samples = [
        "INFO service=auth-api region=eu-central-1 status=200 latency_ms=24 healthcheck passed",
        "INFO service=cache-node region=eu-west-2 processed batch size=42 without errors",
        "INFO service=scheduler region=us-east-1 connection pool steady active=14 idle=4 health=ok",
        "CRITICAL service=worker-pool region=eu-central-1 out of memory oom killer terminated worker",
        "ERROR service=payment-gateway region=us-east-1 status=503 upstream timeout after 4800ms",
        "FATAL service=ingress-proxy region=eu-west-2 disk full write failure on volume /var/lib/data",
    ]
    for text in samples:
        _, body = _request(f"{base}/predict", {"text": text})
        print("PREDICT:", body)

    batch_samples = [
        "INFO service=auth-api region=eu-central-1 status=200 readiness check passed",
        "INFO service=cache-node region=eu-west-2 cache hit ratio stable at 98% errors=0",
        "ERROR service=worker-pool region=us-east-1 tls handshake failed connection refused",
        "CRITICAL service=payment-gateway region=eu-central-1 worker crash loop terminated oom detected",
    ]
    _, batch_raw = _request(f"{base}/predict/batch", {"texts": batch_samples})
    print("BATCH_COUNT:", len(json.loads(batch_raw)))

    _, history_raw = _request(f"{base}/history?limit=10")
    history = json.loads(history_raw)
    print("HISTORY_COUNT:", history.get("count"), "BACKEND:", history.get("storage_backend"))
    if int(history.get("count", 0)) < 10:
        raise RuntimeError(f"Expected >= 10 persisted predictions in /history, got {history}")

    _, drift_raw = _request(f"{base}/drift")
    drift = json.loads(drift_raw)
    print("DRIFT:", drift_raw)
    if int(drift.get("sample_count", 0)) < 10:
        raise RuntimeError(f"Expected >= 10 samples in /drift window, got {drift}")

    _, metrics_raw = _request(f"{base}/metrics")
    if "mlops_inference_requests_total" not in metrics_raw:
        raise RuntimeError("Missing mlops_inference_requests_total in /metrics response")

    return {"ready": ready, "history": history, "drift": drift}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test live MLOps inference service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--expect-backend",
        choices=["sqlite", "postgresql"],
        default=None,
        help="Optional assertion on reported storage_backend from /ready.",
    )
    args = parser.parse_args()
    run_smoke(base_url=args.base_url, expect_backend=args.expect_backend)


if __name__ == "__main__":
    main()
