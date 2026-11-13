from pathlib import Path

from mlops.dataset import build_disjoint_dataset
from mlops.train_export import run_pipeline


def test_disjoint_dataset_is_deterministic_and_balanced():
    train_a, test_a = build_disjoint_dataset(seed=7, train_per_class=40, test_per_class=20)
    train_b, test_b = build_disjoint_dataset(seed=7, train_per_class=40, test_per_class=20)
    assert train_a == train_b
    assert test_a == test_b
    assert len(train_a) == 80
    assert len(test_a) == 40
    assert sum(y for _, y in test_a) == 20


def test_end_to_end_training_onnx_export_and_mlflow(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    mlruns_dir = tmp_path / "mlruns"
    report = run_pipeline(artifacts_dir=artifacts_dir, mlruns_dir=mlruns_dir, seed=42, epochs=4)

    assert (artifacts_dir / "model.onnx").exists()
    assert (artifacts_dir / "model_int8.onnx").exists()
    assert (artifacts_dir / "tokenizer.json").exists()
    assert (artifacts_dir / "baseline_stats.json").exists()
    assert (artifacts_dir / "benchmark_report.json").exists()

    assert report["parity"]["fp32_max_abs_diff"] < 1e-4
    assert report["metrics"]["onnx_int8"]["accuracy"] >= 0.90
    assert mlruns_dir.exists()
