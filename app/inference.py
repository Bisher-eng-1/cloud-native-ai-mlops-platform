from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.tokenizer import TextTokenizer

LABELS = ("NORMAL", "INCIDENT")


@dataclass(frozen=True)
class InferenceResult:
    label: str
    confidence: float
    incident_probability: float
    normal_probability: float
    latency_ms: float
    token_count: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


class ONNXInferenceEngine:

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model_path = model_path
        self.tokenizer = TextTokenizer.load(tokenizer_path)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, text: str) -> InferenceResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[InferenceResult]:
        if not texts:
            return []
        start = time.perf_counter()
        input_ids, attention_mask, token_counts = self.tokenizer.encode_batch(texts)
        (logits,) = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        probs = _softmax(logits)
        total_ms = (time.perf_counter() - start) * 1000.0
        per_item_ms = round(total_ms / len(texts), 3)

        results: list[InferenceResult] = []
        for idx, prob_row in enumerate(probs):
            normal_p = float(prob_row[0])
            incident_p = float(prob_row[1])
            pred_idx = 1 if incident_p >= 0.5 else 0
            label = LABELS[pred_idx]
            confidence = round(max(normal_p, incident_p), 6)
            results.append(
                InferenceResult(
                    label=label,
                    confidence=confidence,
                    incident_probability=round(incident_p, 6),
                    normal_probability=round(normal_p, 6),
                    latency_ms=per_item_ms,
                    token_count=token_counts[idx],
                )
            )
        return results
