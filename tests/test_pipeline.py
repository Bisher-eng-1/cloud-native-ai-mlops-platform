import json
from pathlib import Path

import pytest

from mlops.dataset import build_disjoint_dataset, build_evaluation_splits, load_jsonl_dataset
from mlops.train_export import run_pipeline


def test_disjoint_dataset_is_deterministic_and_balanced():
    train_a, test_a = build_disjoint_dataset(seed=7, train_per_class=40, test_per_class=20)
    train_b, test_b = build_disjoint_dataset(seed=7, train_per_class=40, test_per_class=20)
    assert train_a == train_b
    assert test_a == test_b
    assert len(train_a) == 80
    assert len(test_a) == 40
    assert sum(y for _, y in test_a) == 20

    tr_s, std_s, chl_s = build_evaluation_splits(seed=42, train_per_class=40, test_per_class=120)
    assert len(tr_s) == 80
    assert len(std_s) == 216
    assert len(chl_s) == 24


def test_end_to_end_training_onnx_export_and_mlflow(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    mlruns_dir = tmp_path / "mlruns"
    report = run_pipeline(artifacts_dir=artifacts_dir, mlruns_dir=mlruns_dir, seed=42, epochs=4)

    assert (artifacts_dir / "model.onnx").exists()
    assert (artifacts_dir / "model_int8.onnx").exists()
    assert (artifacts_dir / "tokenizer.json").exists()
    assert (artifacts_dir / "baseline_stats.json").exists()
    assert (artifacts_dir / "benchmark_report.json").exists()

    assert report["dataset"]["source"] == "synthetic_disjoint"
    assert report["dataset"]["challenge_records"] == 24
    assert report["parity"]["fp32_max_abs_diff"] < 1e-4
    assert report["metrics"]["onnx_int8"]["accuracy"] >= 0.90
    assert report["metrics"]["onnx_int8_standard"]["accuracy"] >= 0.90
    assert report["metrics"]["onnx_int8_standard"]["accuracy"] >= report["metrics"]["onnx_int8_challenge"]["accuracy"]
    assert "f1" in report["metrics"]["onnx_int8_challenge"]
    assert mlruns_dir.exists()


def test_load_jsonl_dataset_and_external_pipeline(tmp_path: Path):
    jsonl_path = tmp_path / "logs.jsonl"
    rows = []
    for i in range(12):
        rows.append({"text": f"INFO service=auth region=eu-central-1 status=200 ok seq={i}", "label": 0, "split": "train"})
        rows.append({"text": f"ERROR service=auth region=eu-central-1 status=503 timeout seq={i}", "label": 1, "split": "train"})
    rows.extend(
        [
            {"text": "INFO service=api region=us-east-1 status=200 health=ok", "label": 0, "split": "test"},
            {"text": "ERROR service=api region=us-east-1 status=503 timeout", "label": 1, "split": "test"},
            {"text": "INFO service=api region=eu-west-2 status=200 errors=0", "label": 0},
            {"text": "FATAL service=api region=eu-west-2 disk full write failure", "label": 1},
            {"text": "INFO service=api region=eu-west-2 status=200 passed", "label": 0},
        ]
    )
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")

    train_data, test_data = load_jsonl_dataset(jsonl_path, seed=42)
    assert len(train_data) == 26
    assert len(test_data) == 3

    report = run_pipeline(
        artifacts_dir=tmp_path / "ext_artifacts",
        mlruns_dir=tmp_path / "ext_mlruns",
        seed=42,
        epochs=2,
        dataset_jsonl=jsonl_path,
    )
    assert report["dataset"]["source"] == "external_jsonl"
    assert report["dataset"]["challenge_records"] == 0
    assert report["metrics"]["onnx_int8_challenge"] is None


def test_load_jsonl_dataset_validation_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_jsonl_dataset(tmp_path / "missing.jsonl")

    bad_label = tmp_path / "bad_label.jsonl"
    bad_label.write_text(json.dumps({"text": "INFO ok", "label": 9}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid record"):
        load_jsonl_dataset(bad_label)

    only_train = tmp_path / "only_train.jsonl"
    only_train.write_text(json.dumps({"text": "INFO ok", "label": 0, "split": "train"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one train record"):
        load_jsonl_dataset(only_train)
