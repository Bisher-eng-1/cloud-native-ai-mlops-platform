from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.db import DatabaseManager
from app.drift import DriftDetector, DriftWindow
from app.inference import ONNXInferenceEngine
from app.logging_utils import configure_json_logger
from app.metrics import MLOpsMetrics


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)


class BatchPredictRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(..., min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    label: str
    confidence: float
    incident_probability: float
    normal_probability: float
    latency_ms: float
    token_count: int
    model_version: str
    storage_backend: str


class DriftStatusResponse(BaseModel):
    sample_count: int
    min_samples_required: int
    status: str
    alert: bool
    psi_threshold: float
    input_length_psi: float
    prediction_prob_psi: float
    baseline_mean_length: float
    current_mean_length: float
    baseline_anomaly_rate: float
    current_anomaly_rate: float


class IndexResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    service: str
    model_version: str
    endpoints: list[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    storage_backend: str


class HistoryItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    created_at: str | None
    label: str
    confidence: float
    incident_probability: float
    latency_ms: float
    text_length: int
    text_preview: str
    model_version: str


class HistoryResponse(BaseModel):
    count: int
    storage_backend: str
    items: list[HistoryItem]


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    def _compute_drift(application: FastAPI) -> DriftStatusResponse:
        lengths, probs, labels = application.state.drift_window.snapshot()
        report = application.state.drift.evaluate(lengths, probs, labels)
        application.state.metrics.input_length_psi.set(report.input_length_psi)
        application.state.metrics.prediction_prob_psi.set(report.prediction_prob_psi)
        application.state.metrics.drift_alert.set(1 if report.alert else 0)
        return DriftStatusResponse(
            sample_count=report.sample_count,
            min_samples_required=report.min_samples_required,
            status=report.status,
            alert=report.alert,
            psi_threshold=application.state.settings.drift_psi_threshold,
            input_length_psi=report.input_length_psi,
            prediction_prob_psi=report.prediction_prob_psi,
            baseline_mean_length=report.baseline_mean_length,
            current_mean_length=report.current_mean_length,
            baseline_anomaly_rate=report.baseline_anomaly_rate,
            current_anomaly_rate=report.current_anomaly_rate,
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        db = DatabaseManager(cfg.database_url)
        db.init_tables()
        engine = ONNXInferenceEngine(cfg.model_file, cfg.tokenizer_file)
        drift_detector = DriftDetector.from_file(
            cfg.baseline_stats_file,
            psi_threshold=cfg.drift_psi_threshold,
            min_samples=10,
        )
        drift_window = DriftWindow(max_size=cfg.drift_window_size)
        seed_rows = list(reversed(db.recent_predictions(limit=cfg.drift_window_size)))
        drift_window.seed(
            [(row.text_length, row.anomaly_probability, row.predicted_label) for row in seed_rows]
        )
        metrics = MLOpsMetrics()
        metrics.model_loaded.set(1)

        application.state.settings = cfg
        application.state.db = db
        application.state.engine = engine
        application.state.drift = drift_detector
        application.state.drift_window = drift_window
        application.state.metrics = metrics
        _compute_drift(application)

        try:
            yield
        finally:
            metrics.model_loaded.set(0)
            db.dispose()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.model_version,
        description="ONNX-accelerated NLP inference API with PostgreSQL logging, PSI drift checks, and Prometheus metrics.",
        lifespan=lifespan,
    )
    logger = configure_json_logger()

    @app.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @app.get("/", response_model=IndexResponse)
    def index() -> IndexResponse:
        return IndexResponse(
            service=cfg.app_name,
            model_version=cfg.model_version,
            endpoints=["/health", "/ready", "/predict", "/predict/batch", "/drift", "/history", "/metrics", "/docs"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="healthy", model_version=app.state.settings.model_version)

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        if getattr(app.state, "engine", None) is None or not app.state.db.ping():
            raise HTTPException(status_code=503, detail="Database or model readiness check failed")
        return ReadyResponse(
            status="ready",
            model_version=app.state.settings.model_version,
            storage_backend=app.state.db.backend_kind(),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictRequest) -> PredictionResponse:
        clean_text = payload.text.strip()
        if not clean_text:
            raise HTTPException(status_code=422, detail="Input text cannot be whitespace only")

        result = app.state.engine.predict(clean_text)
        record = app.state.db.record_prediction(
            text_input=clean_text,
            token_count=result.token_count,
            predicted_label=result.label,
            confidence=result.confidence,
            anomaly_probability=result.incident_probability,
            latency_ms=result.latency_ms,
            model_version=app.state.settings.model_version,
        )

        app.state.metrics.requests_total.labels(label=result.label, status="200").inc()
        app.state.metrics.inference_latency_seconds.observe(result.latency_ms / 1000.0)
        app.state.metrics.prediction_confidence.observe(result.confidence)
        app.state.drift_window.append(len(clean_text), result.incident_probability, result.label)
        _compute_drift(app)

        return PredictionResponse(
            id=record.id,
            label=result.label,
            confidence=result.confidence,
            incident_probability=result.incident_probability,
            normal_probability=result.normal_probability,
            latency_ms=result.latency_ms,
            token_count=result.token_count,
            model_version=app.state.settings.model_version,
            storage_backend=app.state.db.backend_kind(),
        )

    @app.post("/predict/batch", response_model=list[PredictionResponse])
    def predict_batch(payload: BatchPredictRequest) -> list[PredictionResponse]:
        cleaned = [item.strip() for item in payload.texts]
        if any(not item for item in cleaned):
            raise HTTPException(status_code=422, detail="Batch entries cannot be whitespace only")

        results = app.state.engine.predict_batch(cleaned)
        batch_payloads = [
            {
                "text_input": text_item,
                "token_count": res.token_count,
                "predicted_label": res.label,
                "confidence": res.confidence,
                "anomaly_probability": res.incident_probability,
                "latency_ms": res.latency_ms,
                "model_version": app.state.settings.model_version,
                "request_source": "api_batch",
            }
            for text_item, res in zip(cleaned, results, strict=True)
        ]
        records = app.state.db.record_predictions_batch(batch_payloads)

        responses: list[PredictionResponse] = []
        window_entries: list[tuple[int, float, str]] = []
        backend = app.state.db.backend_kind()
        for text_item, record, res in zip(cleaned, records, results, strict=True):
            app.state.metrics.requests_total.labels(label=res.label, status="200").inc()
            app.state.metrics.inference_latency_seconds.observe(res.latency_ms / 1000.0)
            app.state.metrics.prediction_confidence.observe(res.confidence)
            window_entries.append((len(text_item), res.incident_probability, res.label))
            responses.append(
                PredictionResponse(
                    id=record.id,
                    label=res.label,
                    confidence=res.confidence,
                    incident_probability=res.incident_probability,
                    normal_probability=res.normal_probability,
                    latency_ms=res.latency_ms,
                    token_count=res.token_count,
                    model_version=app.state.settings.model_version,
                    storage_backend=backend,
                )
            )

        app.state.drift_window.extend(window_entries)
        _compute_drift(app)
        return responses

    @app.get("/drift", response_model=DriftStatusResponse)
    def drift_status() -> DriftStatusResponse:
        return _compute_drift(app)

    @app.get("/history", response_model=HistoryResponse)
    def history(limit: int = Query(default=20, ge=1, le=200)) -> HistoryResponse:
        rows = app.state.db.recent_predictions(limit=limit)
        return HistoryResponse(
            count=len(rows),
            storage_backend=app.state.db.backend_kind(),
            items=[
                HistoryItem(
                    id=row.id,
                    created_at=row.created_at.isoformat() if row.created_at else None,
                    label=row.predicted_label,
                    confidence=row.confidence,
                    incident_probability=row.anomaly_probability,
                    latency_ms=row.latency_ms,
                    text_length=row.text_length,
                    text_preview=row.text_preview,
                    model_version=row.model_version,
                )
                for row in rows
            ],
        )

    @app.get("/metrics")
    def metrics() -> Response:
        body, content_type = app.state.metrics.render()
        return Response(content=body, media_type=content_type)

    return app


app = create_app()
