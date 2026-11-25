from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_PATH = _ROOT / "scripts" / "smoke_demo.py"
_SPEC = importlib.util.spec_from_file_location("smoke_demo", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_demo)


def test_run_smoke_verifies_backend_history_drift_and_metrics(monkeypatch):
    calls: list[str] = []

    def fake_request(url: str, payload=None):
        calls.append(url)
        if url.endswith("/ready"):
            return 200, json.dumps({"status": "ready", "storage_backend": "postgresql"})
        if url.endswith("/predict"):
            return 200, json.dumps({"label": "NORMAL"})
        if url.endswith("/predict/batch"):
            return 200, json.dumps([{"label": "NORMAL"}] * 4)
        if "/history" in url:
            return 200, json.dumps({"count": 10, "storage_backend": "postgresql", "items": []})
        if url.endswith("/drift"):
            return 200, json.dumps({"sample_count": 10, "status": "HEALTHY", "prediction_prob_psi": 0.02})
        if url.endswith("/metrics"):
            return 200, "# TYPE mlops_inference_requests_total counter\nmlops_inference_requests_total 10\n"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(smoke_demo, "_request", fake_request)
    summary = smoke_demo.run_smoke("http://127.0.0.1:8000/", expect_backend="postgresql")
    assert summary["ready"]["storage_backend"] == "postgresql"
    assert summary["history"]["count"] == 10
    assert summary["drift"]["sample_count"] == 10
    assert len(calls) == 11


def test_run_smoke_rejects_mismatched_storage_backend(monkeypatch):
    def fake_ready(url: str, payload=None):
        return 200, json.dumps({"status": "ready", "storage_backend": "sqlite"})

    monkeypatch.setattr(smoke_demo, "_request", fake_ready)
    with pytest.raises(RuntimeError, match="Expected storage_backend='postgresql'"):
        smoke_demo.run_smoke("http://127.0.0.1:8000", expect_backend="postgresql")


def test_k8s_manifests_wire_shared_postgres_and_configmap():
    k8s_dir = _ROOT / "k8s"
    configmap = (k8s_dir / "configmap.yaml").read_text(encoding="utf-8")
    postgres = (k8s_dir / "postgres.yaml").read_text(encoding="utf-8")
    deployment = (k8s_dir / "deployment.yaml").read_text(encoding="utf-8")

    assert "mlops-inference-config" in configmap
    assert "ORT_INTRA_OP_THREADS" in configmap
    assert "kind: Service" in postgres and "mlops-postgres" in postgres
    assert "configMapRef" in deployment and "secretKeyRef" in deployment
    assert "wait-for-postgres" in deployment
