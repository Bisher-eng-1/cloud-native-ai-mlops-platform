# Cloud-Native AI Inference & MLOps Platform

FastAPI inference service running an INT8-quantized ONNX Transformer classifier (`LightweightLogTransformer`) with SQLite/PostgreSQL prediction logging, in-memory PSI drift tracking, Prometheus/Grafana metrics, Kubernetes manifests, and Terraform IaC.

## Architecture

- **Training & Export (`mlops/`)**: trains a 2-layer PyTorch encoder on synthetic operational logs, exports to ONNX FP32, applies dynamic INT8 quantization (`artifacts/model_int8.onnx`), benchmarks PyTorch vs ONNX latency, and logs metrics/artifacts to MLflow.
- **Serving (`app/`)**: multi-stage Python 3.11 runtime with `onnxruntime` only (no PyTorch in the serving container). Persists predictions via SQLAlchemy (`postgresql+psycopg` or SQLite fallback) and computes Population Stability Index (PSI) from an in-memory sliding window (`DriftWindow`) seeded from the database on startup.
- **Observability & Infra**: exposes `/metrics` (`mlops_inference_requests_total`, `mlops_inference_latency_seconds`, `mlops_input_length_psi`, `mlops_prediction_prob_psi`, `mlops_drift_alert_active`), Prometheus + Grafana provisioning under `monitoring/`, Kubernetes `Deployment` + `HPA` under `k8s/`, and Cloud Run / Artifact Registry Terraform under `infra/terraform/`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe and active model version |
| `GET` | `/ready` | Verifies ONNX session and database connectivity |
| `POST` | `/predict` | Single log classification (`NORMAL` / `INCIDENT`) |
| `POST` | `/predict/batch` | Batch classification (up to 64 items) |
| `GET` | `/drift` | Current sliding-window PSI status vs training baseline |
| `GET` | `/history?limit=20` | Recent persisted predictions |
| `GET` | `/metrics` | Prometheus text exposition |

## Running Locally

Export the model and run tests:

```bash
python -m mlops.train_export --epochs 8
pytest -v --cov=app --cov=mlops --cov-fail-under=85
```

Start the API with Docker Compose (API `:8000`, Postgres `:5432`, Prometheus `:9090`, Grafana `:3001`):

```bash
docker compose up --build
```

Verify Terraform formatting and config:

```bash
cd infra/terraform && terraform fmt -check && terraform init -backend=false && terraform validate
```

## Limitations

- The classifier is trained on synthetic template logs (`mlops/dataset.py`); accuracy numbers (`1.0` on the held-out synthetic split) reflect that controlled corpus rather than arbitrary production logs.
- `DriftWindow` lives in process memory per replica (seeded once at startup from the latest `DRIFT_WINDOW_SIZE` rows), so multi-pod deployments compute PSI per pod window rather than globally across replicas.
- `scripts/smoke_demo.py` assumes the inference service is already running on `--base-url`.
