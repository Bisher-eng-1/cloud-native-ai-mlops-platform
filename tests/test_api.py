from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import normalize_database_url
from app.main import create_app


def _make_test_client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "test_predictions.db"
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        model_path="artifacts/model_int8.onnx",
        tokenizer_path="artifacts/tokenizer.json",
        baseline_stats_path="artifacts/baseline_stats.json",
        drift_window_size=50,
        drift_psi_threshold=0.25,
    )
    app = create_app(settings=settings)
    return TestClient(app)


def test_normalize_postgres_urls():
    assert normalize_database_url("postgres://u:p@host/db").startswith("postgresql+psycopg://")
    assert normalize_database_url("postgresql://u:p@host/db").startswith("postgresql+psycopg://")
    assert normalize_database_url("sqlite:///./local.db") == "sqlite:///./local.db"


def test_health_and_ready_endpoints(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "/predict" in root.json()["endpoints"]
        assert "X-Request-ID" in root.headers

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["storage_backend"] == "sqlite"


def test_predict_and_history_persistence(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        r_norm = client.post(
            "/predict",
            json={"text": "INFO service=auth-api region=eu-central-1 status=200 latency_ms=25 healthcheck passed"},
        )
        assert r_norm.status_code == 200
        body_norm = r_norm.json()
        assert body_norm["label"] == "NORMAL"
        assert 0.0 <= body_norm["confidence"] <= 1.0

        r_inc = client.post(
            "/predict",
            json={"text": "CRITICAL service=worker-pool region=eu-west-2 out of memory oom killer terminated worker"},
        )
        assert r_inc.status_code == 200
        body_inc = r_inc.json()
        assert body_inc["label"] == "INCIDENT"
        assert body_inc["incident_probability"] >= 0.5

        hist = client.get("/history?limit=10")
        assert hist.status_code == 200
        assert hist.json()["count"] == 2


def test_batch_predict_and_validation(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        bad = client.post("/predict", json={"text": "   "})
        assert bad.status_code == 422

        batch = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "INFO service=cache-node region=us-east-1 readiness probe ok response_time=19ms",
                    "ERROR service=payment-gateway region=eu-central-1 status=503 upstream timeout after 4200ms",
                ]
            },
        )
        assert batch.status_code == 200
        items = batch.json()
        assert len(items) == 2
        assert items[0]["label"] == "NORMAL"
        assert items[1]["label"] == "INCIDENT"


def test_drift_and_prometheus_metrics(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        texts = [
            "CRITICAL service=worker-pool region=eu-central-1 out of memory oom killer terminated worker " + ("x" * 180)
            for _ in range(12)
        ]
        res = client.post("/predict/batch", json={"texts": texts})
        assert res.status_code == 200

        drift = client.get("/drift")
        assert drift.status_code == 200
        d = drift.json()
        assert d["sample_count"] == 12
        assert d["alert"] is True
        assert d["status"] == "DRIFT_DETECTED"

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        text_body = metrics.text
        assert "mlops_inference_requests_total" in text_body
        assert "mlops_inference_latency_seconds" in text_body
        assert "mlops_input_length_psi" in text_body
        assert "mlops_drift_alert_active 1.0" in text_body


def test_hot_path_drift_uses_memory_window_without_db_reads(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        def _fail_db_read(*_args, **_kwargs):
            raise AssertionError("recent_predictions should not run on inference hot path")

        client.app.state.db.recent_predictions = _fail_db_read

        r1 = client.post("/predict", json={"text": "INFO service=auth-api status=200 healthcheck passed"})
        assert r1.status_code == 200

        r2 = client.post(
            "/predict/batch",
            json={"texts": ["INFO service=cache-node status=200 ok", "ERROR service=gateway status=503 timeout"]},
        )
        assert r2.status_code == 200

        drift = client.get("/drift")
        assert drift.status_code == 200
        assert drift.json()["sample_count"] == 3


def test_ready_failure_batch_whitespace_and_db_session(tmp_path: Path):
    with _make_test_client(tmp_path) as client:
        bad_batch = client.post("/predict/batch", json={"texts": ["INFO ok", "   "]})
        assert bad_batch.status_code == 422

        db_gen = client.app.state.db.session()
        session = next(db_gen)
        assert session is not None
        db_gen.close()

        original_ping = client.app.state.db.ping
        client.app.state.db.ping = lambda: False
        try:
            res = client.get("/ready")
            assert res.status_code == 503
        finally:
            client.app.state.db.ping = original_ping
