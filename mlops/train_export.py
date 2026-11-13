from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from app.tokenizer import TextTokenizer
from mlops.dataset import build_disjoint_dataset
from mlops.model import LightweightLogTransformer


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _benchmark_pytorch(model: nn.Module, ids_np: np.ndarray, mask_np: np.ndarray, iterations: int = 120) -> dict[str, float]:
    model.eval()
    sample_ids = torch.from_numpy(ids_np[:1])
    sample_mask = torch.from_numpy(mask_np[:1])
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_ids, sample_mask)
        latencies = []
        for idx in range(iterations):
            row = idx % len(ids_np)
            t_ids = torch.from_numpy(ids_np[row : row + 1])
            t_mask = torch.from_numpy(mask_np[row : row + 1])
            start = time.perf_counter()
            _ = model(t_ids, t_mask)
            latencies.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
    }


def _benchmark_onnx(onnx_path: Path, ids_np: np.ndarray, mask_np: np.ndarray, iterations: int = 120) -> tuple[dict[str, float], np.ndarray]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    for _ in range(10):
        session.run(["logits"], {"input_ids": ids_np[:1], "attention_mask": mask_np[:1]})

    latencies = []
    for idx in range(iterations):
        row = idx % len(ids_np)
        start = time.perf_counter()
        session.run(["logits"], {"input_ids": ids_np[row : row + 1], "attention_mask": mask_np[row : row + 1]})
        latencies.append((time.perf_counter() - start) * 1000.0)

    (all_logits,) = session.run(["logits"], {"input_ids": ids_np, "attention_mask": mask_np})
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
    }, all_logits


def _build_baseline_stats(texts: list[str], probs: np.ndarray) -> dict:
    lengths = np.asarray([len(t) for t in texts], dtype=np.float64)
    incident_probs = probs[:, 1]

    len_edges = [0.0, 60.0, 80.0, 100.0, 130.0, 10000.0]
    len_counts, _ = np.histogram(lengths, bins=len_edges)
    len_probs = [round(max(float(c) / len(lengths), 1e-4), 6) for c in len_counts]

    prob_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    prob_counts, _ = np.histogram(incident_probs, bins=prob_edges)
    prob_probs = [round(max(float(c) / len(incident_probs), 1e-4), 6) for c in prob_counts]

    return {
        "sample_size": len(texts),
        "mean_text_length": round(float(np.mean(lengths)), 2),
        "anomaly_rate": round(float(np.mean(incident_probs >= 0.5)), 4),
        "length_bin_edges": len_edges,
        "length_bin_probs": len_probs,
        "prob_bin_edges": prob_edges,
        "prob_bin_probs": prob_probs,
    }


def run_pipeline(
    artifacts_dir: Path = Path("artifacts"),
    mlruns_dir: Path = Path("mlruns"),
    seed: int = 42,
    epochs: int = 10,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = build_disjoint_dataset(seed=seed)
    train_texts = [t for t, _ in train_data]
    train_labels = [y for _, y in train_data]
    test_texts = [t for t, _ in test_data]
    test_labels = [y for _, y in test_data]

    tokenizer = TextTokenizer.fit(train_texts, max_vocab_size=2048, max_seq_len=48)
    tokenizer_path = artifacts_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    tr_ids, tr_mask, _ = tokenizer.encode_batch(train_texts)
    te_ids, te_mask, _ = tokenizer.encode_batch(test_texts)

    dataset = TensorDataset(
        torch.from_numpy(tr_ids),
        torch.from_numpy(tr_mask),
        torch.tensor(train_labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LightweightLogTransformer(vocab_size=len(tokenizer.vocab), max_seq_len=tokenizer.max_seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_ids, batch_mask, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_ids, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pt_test_logits = model(torch.from_numpy(te_ids), torch.from_numpy(te_mask)).numpy()
        pt_train_logits = model(torch.from_numpy(tr_ids), torch.from_numpy(tr_mask)).numpy()

    pt_preds = np.argmax(pt_test_logits, axis=1).tolist()
    pt_metrics = _compute_metrics(test_labels, pt_preds)

    fp32_onnx_path = artifacts_dir / "model.onnx"
    int8_onnx_path = artifacts_dir / "model_int8.onnx"

    dummy_ids = torch.from_numpy(te_ids[:1])
    dummy_mask = torch.from_numpy(te_mask[:1])
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(fp32_onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )

    pre_onnx_path = artifacts_dir / "model_prep.onnx"
    quant_pre_process(
        input_model=str(fp32_onnx_path),
        output_model_path=str(pre_onnx_path),
        skip_symbolic_shape=True,
    )
    quantize_dynamic(
        model_input=str(pre_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    pre_onnx_path.unlink(missing_ok=True)

    pt_bench = _benchmark_pytorch(model, te_ids, te_mask)
    ort_fp32_bench, ort_fp32_logits = _benchmark_onnx(fp32_onnx_path, te_ids, te_mask)
    ort_int8_bench, ort_int8_logits = _benchmark_onnx(int8_onnx_path, te_ids, te_mask)

    max_abs_diff_fp32 = float(np.max(np.abs(pt_test_logits - ort_fp32_logits)))
    max_abs_diff_int8 = float(np.max(np.abs(pt_test_logits - ort_int8_logits)))
    ort_fp32_metrics = _compute_metrics(test_labels, np.argmax(ort_fp32_logits, axis=1).tolist())
    ort_int8_metrics = _compute_metrics(test_labels, np.argmax(ort_int8_logits, axis=1).tolist())

    fp32_kb = round(fp32_onnx_path.stat().st_size / 1024.0, 2)
    int8_kb = round(int8_onnx_path.stat().st_size / 1024.0, 2)
    speedup_int8 = round(pt_bench["mean_ms"] / max(ort_int8_bench["mean_ms"], 1e-6), 2)

    baseline_stats = _build_baseline_stats(train_texts, _softmax(pt_train_logits))
    baseline_path = artifacts_dir / "baseline_stats.json"
    baseline_path.write_text(json.dumps(baseline_stats, indent=2), encoding="utf-8")

    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "mlflow": mlflow.__version__,
        },
        "dataset": {
            "train_records": len(train_texts),
            "test_records": len(test_texts),
            "vocab_size": len(tokenizer.vocab),
        },
        "parity": {
            "fp32_max_abs_diff": round(max_abs_diff_fp32, 7),
            "int8_max_abs_diff": round(max_abs_diff_int8, 7),
        },
        "model_size_kb": {
            "onnx_fp32": fp32_kb,
            "onnx_int8": int8_kb,
            "compression_ratio": round(fp32_kb / max(int8_kb, 1e-6), 2),
        },
        "metrics": {
            "pytorch_fp32": pt_metrics,
            "onnx_fp32": ort_fp32_metrics,
            "onnx_int8": ort_int8_metrics,
        },
        "latency_benchmark": {
            "pytorch_fp32": pt_bench,
            "onnx_fp32": ort_fp32_bench,
            "onnx_int8": ort_int8_bench,
            "onnx_int8_speedup_vs_pytorch": speedup_int8,
        },
    }
    report_path = artifacts_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("cloud-native-nlp-transformer-onnx")
    with mlflow.start_run(run_name="transformer-onnx-int8"):
        mlflow.log_params(
            {
                "seed": seed,
                "epochs": epochs,
                "vocab_size": len(tokenizer.vocab),
                "max_seq_len": tokenizer.max_seq_len,
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "quantization": "dynamic_int8",
            }
        )
        mlflow.log_metrics(
            {
                "pytorch_accuracy": pt_metrics["accuracy"],
                "pytorch_f1": pt_metrics["f1"],
                "onnx_fp32_accuracy": ort_fp32_metrics["accuracy"],
                "onnx_int8_accuracy": ort_int8_metrics["accuracy"],
                "onnx_int8_f1": ort_int8_metrics["f1"],
                "fp32_max_abs_diff": max_abs_diff_fp32 if math.isfinite(max_abs_diff_fp32) else 0.0,
                "pytorch_mean_ms": pt_bench["mean_ms"],
                "onnx_fp32_mean_ms": ort_fp32_bench["mean_ms"],
                "onnx_int8_mean_ms": ort_int8_bench["mean_ms"],
                "onnx_int8_speedup": speedup_int8,
                "onnx_fp32_kb": fp32_kb,
                "onnx_int8_kb": int8_kb,
            }
        )
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(baseline_path))
        mlflow.log_artifact(str(int8_onnx_path))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer classifier, export ONNX INT8, benchmark, and log to MLflow.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mlruns-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    report = run_pipeline(artifacts_dir=args.artifacts_dir, mlruns_dir=args.mlruns_dir, epochs=args.epochs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
