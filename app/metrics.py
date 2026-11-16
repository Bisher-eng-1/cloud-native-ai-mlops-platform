from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class MLOpsMetrics:

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests_total = Counter(
            "mlops_inference_requests_total",
            "Total inference requests served by label and HTTP status",
            labelnames=["label", "status"],
            registry=self.registry,
        )
        self.inference_latency_seconds = Histogram(
            "mlops_inference_latency_seconds",
            "ONNX Runtime inference latency in seconds",
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry,
        )
        self.prediction_confidence = Histogram(
            "mlops_prediction_confidence",
            "Model prediction confidence distribution",
            buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )
        self.input_length_psi = Gauge(
            "mlops_input_length_psi",
            "Population Stability Index for input text length vs training baseline",
            registry=self.registry,
        )
        self.prediction_prob_psi = Gauge(
            "mlops_prediction_prob_psi",
            "Population Stability Index for anomaly probability vs training baseline",
            registry=self.registry,
        )
        self.drift_alert = Gauge(
            "mlops_drift_alert_active",
            "1 if input or prediction PSI exceeds configured threshold else 0",
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "mlops_model_loaded",
            "1 if ONNX model session is initialized and ready",
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
