# Architecture Decision Records (ADR)

## ADR-001: Separate Training (`torch`) and Serving (`onnxruntime`) Dependencies
- **Context:** Installing full PyTorch (~180 MB–2+ GB depending on platform) inside serving containers slows Docker builds, inflates image attack surface, and consumes unnecessary memory on `8 GB RAM` development laptops.
- **Decision:** Keep PyTorch + MLflow strictly in `requirements-mlops.txt` for offline/CI training and ONNX export, while the production `Dockerfile` installs only `requirements-serve.txt` (`onnxruntime`, `fastapi`, `sqlalchemy`, `prometheus-client`).
- **Consequence:** Serving containers stay small, build quickly, and achieve lower inference latency via ONNX Runtime dynamic INT8 quantization.

## ADR-002: Dynamic INT8 Quantization for Transformer Inference
- **Context:** CPU-bound inference pods scale horizontally on CPU utilization (`HPA`). Reducing matrix multiplication weight width from FP32 to INT8 lowers memory footprint and improves cache locality on CPU.
- **Decision:** Export the Transformer Encoder via TorchScript ONNX (`opset 14`), run `quant_pre_process`, and apply dynamic `QInt8` quantization, logging both numerical parity (`max_abs_diff`) and held-out F1 metrics to MLflow.
- **Consequence:** Evaluated directly against PyTorch FP32 in `artifacts/benchmark_report.json` so any accuracy/latency trade-off is transparent and reproducible.

## ADR-003: Population Stability Index (PSI) for Drift Detection
- **Context:** Production traffic can change in text length or prediction confidence distribution before ground-truth labels become available.
- **Decision:** Store baseline training length and probability histograms in `artifacts/baseline_stats.json` and evaluate sliding windows from PostgreSQL/SQLite using PSI (`threshold = 0.25`), exporting live gauges to Prometheus (`/metrics`).
- **Consequence:** Enables zero-label early alerting in Grafana without adding heavy external drift-worker infrastructure.

## ADR-004: SQLAlchemy Backend Normalization for Neon Cloud PostgreSQL
- **Context:** Developers often copy raw `postgres://` or `postgresql://` URLs from Neon/Supabase dashboards while SQLAlchemy + `psycopg` v3 expects `postgresql+psycopg://`.
- **Decision:** Normalize connection strings in `app/db.py` (`normalize_database_url`) with SQLite fallback for isolated unit testing.
- **Consequence:** Zero-change portability between local SQLite, Compose PostgreSQL, and Neon Cloud Free Tier.
