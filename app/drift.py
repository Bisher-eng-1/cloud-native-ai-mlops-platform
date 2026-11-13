from __future__ import annotations

import json
import math
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
    """Compute Population Stability Index (PSI) between baseline proportions and current values."""
    if not observed_values:
        return 0.0
    counts, _ = np.histogram(np.asarray(observed_values, dtype=np.float64), bins=np.asarray(bin_edges, dtype=np.float64))
    total = float(counts.sum())
    if total <= 0:
        return 0.0

    psi = 0.0
    for exp_p, obs_count in zip(expected_probs, counts, strict=False):
        actual_p = max(float(obs_count) / total, EPSILON)
        expected_p = max(float(exp_p), EPSILON)
        psi += (actual_p - expected_p) * math.log(actual_p / expected_p)
    return round(float(psi), 6)


class DriftDetector:
    """Compares recent production inference windows against training baseline histograms."""

    def __init__(self, baseline_stats: dict, psi_threshold: float = 0.25, min_samples: int = 10) -> None:
        self.baseline = baseline_stats
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples

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
