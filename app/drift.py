from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPSILON = 1e-4


@dataclass(frozen=True)
class DriftReport:
    sample_count: int
    min_samples_required: int
    status: str
    alert: bool
    input_length_psi: float
    prediction_prob_psi: float
    baseline_mean_length: float
    current_mean_length: float
    baseline_anomaly_rate: float
    current_anomaly_rate: float


def compute_psi(expected_probs: list[float], observed_values: list[float], bin_edges: list[float]) -> float:
    if not observed_values:
        return 0.0
    edges = np.asarray(bin_edges, dtype=np.float64).copy()
    if edges.size >= 2:
        edges[0] = -np.inf
        edges[-1] = np.inf
    counts, _ = np.histogram(np.asarray(observed_values, dtype=np.float64), bins=edges)
    total = float(counts.sum())
    if total <= 0:
        return 0.0

    psi = 0.0
    for exp_p, obs_count in zip(expected_probs, counts, strict=False):
        actual_p = max(float(obs_count) / total, EPSILON)
        expected_p = max(float(exp_p), EPSILON)
        psi += (actual_p - expected_p) * math.log(actual_p / expected_p)
    return round(float(psi), 6)


class DriftWindow:
    def __init__(self, max_size: int) -> None:
        self._lock = threading.Lock()
        self._samples: deque[tuple[int, float, str]] = deque(maxlen=max_size)

    def seed(self, samples: list[tuple[int, float, str]]) -> None:
        with self._lock:
            self._samples.clear()
            self._samples.extend(samples)

    def append(self, length: int, anomaly_prob: float, label: str) -> None:
        with self._lock:
            self._samples.append((length, anomaly_prob, label))

    def extend(self, samples: list[tuple[int, float, str]]) -> None:
        with self._lock:
            self._samples.extend(samples)

    def snapshot(self) -> tuple[list[int], list[float], list[str]]:
        with self._lock:
            items = list(self._samples)
        if not items:
            return [], [], []
        lengths, probs, labels = zip(*items, strict=True)
        return list(lengths), list(probs), list(labels)


class DriftDetector:
    def __init__(self, baseline_stats: dict, psi_threshold: float = 0.25, min_samples: int = 10) -> None:
        self.baseline = baseline_stats
        self.psi_threshold = psi_threshold
        self.min_samples = max(1, min_samples)

    @classmethod
    def from_file(cls, path: Path, psi_threshold: float = 0.25, min_samples: int = 10) -> DriftDetector:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(baseline_stats=data, psi_threshold=psi_threshold, min_samples=min_samples)

    def evaluate(self, lengths: list[int], anomaly_probs: list[float], predicted_labels: list[str]) -> DriftReport:
        count = len(lengths)
        baseline_len = float(self.baseline["mean_text_length"])
        baseline_anom_rate = float(self.baseline["anomaly_rate"])

        if count == 0:
            return DriftReport(
                sample_count=0,
                min_samples_required=self.min_samples,
                status="INSUFFICIENT_DATA",
                alert=False,
                input_length_psi=0.0,
                prediction_prob_psi=0.0,
                baseline_mean_length=baseline_len,
                current_mean_length=0.0,
                baseline_anomaly_rate=baseline_anom_rate,
                current_anomaly_rate=0.0,
            )

        current_len = round(sum(lengths) / count, 2)
        current_anom_rate = round(
            sum(1 for label in predicted_labels if label == "INCIDENT") / count,
            4,
        )

        if count < self.min_samples:
            return DriftReport(
                sample_count=count,
                min_samples_required=self.min_samples,
                status="COLLECTING_BASELINE_WINDOW",
                alert=False,
                input_length_psi=0.0,
                prediction_prob_psi=0.0,
                baseline_mean_length=baseline_len,
                current_mean_length=current_len,
                baseline_anomaly_rate=baseline_anom_rate,
                current_anomaly_rate=current_anom_rate,
            )

        len_psi = compute_psi(
            self.baseline["length_bin_probs"],
            [float(v) for v in lengths],
            self.baseline["length_bin_edges"],
        )
        prob_psi = compute_psi(
            self.baseline["prob_bin_probs"],
            [float(p) for p in anomaly_probs],
            self.baseline["prob_bin_edges"],
        )
        alert = (len_psi >= self.psi_threshold) or (prob_psi >= self.psi_threshold)
        status = "DRIFT_DETECTED" if alert else "OK"

        return DriftReport(
            sample_count=count,
            min_samples_required=self.min_samples,
            status=status,
            alert=alert,
            input_length_psi=len_psi,
            prediction_prob_psi=prob_psi,
            baseline_mean_length=baseline_len,
            current_mean_length=current_len,
            baseline_anomaly_rate=baseline_anom_rate,
            current_anomaly_rate=current_anom_rate,
        )
