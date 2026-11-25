from pathlib import Path

import pytest

from app.drift import DriftDetector, DriftWindow, compute_psi
from app.inference import ONNXInferenceEngine


def test_onnx_engine_predicts_accurately():
    engine = ONNXInferenceEngine(
        model_path=Path("artifacts/model_int8.onnx"),
        tokenizer_path=Path("artifacts/tokenizer.json"),
    )
    normal = engine.predict("INFO service=auth-api region=eu-west-2 status=200 latency_ms=22 healthcheck passed")
    incident = engine.predict("FATAL service=cache-node region=eu-central-1 disk full write failure on volume")
    assert normal.label == "NORMAL"
    assert incident.label == "INCIDENT"
    assert abs((normal.normal_probability + normal.incident_probability) - 1.0) < 1e-4
    assert engine.predict_batch([]) == []


def test_onnx_engine_threads_and_warmup():
    engine = ONNXInferenceEngine(
        model_path=Path("artifacts/model_int8.onnx"),
        tokenizer_path=Path("artifacts/tokenizer.json"),
        intra_op_threads=1,
        inter_op_threads=1,
        warmup_runs=2,
    )
    assert engine.intra_op_threads == 1
    assert engine.inter_op_threads == 1
    assert engine.warmup_runs_completed == 2
    assert engine.warmup_ms >= 0.0
    assert engine.warmup(0) == 0.0
    assert engine.warmup(1) >= 0.0
    assert engine.warmup_runs_completed == 3


def test_onnx_engine_missing_artifacts_raise(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ONNXInferenceEngine(model_path=tmp_path / "missing.onnx", tokenizer_path=Path("artifacts/tokenizer.json"))
    with pytest.raises(FileNotFoundError):
        ONNXInferenceEngine(model_path=Path("artifacts/model_int8.onnx"), tokenizer_path=tmp_path / "missing.json")


def test_psi_is_near_zero_on_baseline_distribution():
    detector = DriftDetector.from_file(Path("artifacts/baseline_stats.json"), psi_threshold=0.25, min_samples=10)
    empty_report = detector.evaluate([], [], [])
    assert empty_report.status == "INSUFFICIENT_DATA"
    assert empty_report.alert is False

    warm_report = detector.evaluate([70, 72], [0.1, 0.9], ["NORMAL", "INCIDENT"])
    assert warm_report.status == "COLLECTING_BASELINE_WINDOW"
    assert warm_report.alert is False

    lengths = [75] * 20
    probs = [0.05] * 10 + [0.95] * 10
    labels = ["NORMAL"] * 10 + ["INCIDENT"] * 10
    report = detector.evaluate(lengths, probs, labels)
    assert report.sample_count == 20
    assert report.prediction_prob_psi < 0.25


def test_psi_detects_severe_shift_and_out_of_range_values():
    expected = [0.25, 0.25, 0.25, 0.25]
    edges = [0.0, 0.25, 0.5, 0.75, 1.01]
    shifted = [0.98] * 40
    psi = compute_psi(expected, shifted, edges)
    assert psi > 1.0
    assert compute_psi(expected, [], edges) == 0.0

    extreme_out_of_range = [50000.0] * 20
    assert compute_psi(expected, extreme_out_of_range, edges) > 1.0


def test_drift_window_caps_at_maxlen_and_preserves_order():
    win = DriftWindow(max_size=3)
    assert win.snapshot() == ([], [], [])
    win.seed([(10, 0.1, "NORMAL"), (20, 0.2, "NORMAL")])
    win.append(30, 0.8, "INCIDENT")
    win.extend([(40, 0.9, "INCIDENT")])
    lengths, probs, labels = win.snapshot()
    assert lengths == [20, 30, 40]
    assert probs == [0.2, 0.8, 0.9]
    assert labels == ["NORMAL", "INCIDENT", "INCIDENT"]
