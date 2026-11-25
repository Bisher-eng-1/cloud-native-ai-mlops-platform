from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "Cloud-Native AI Inference & MLOps Platform"
    model_version: str = "transformer-onnx-int8-v1"
    model_path: str = "artifacts/model_int8.onnx"
    tokenizer_path: str = "artifacts/tokenizer.json"
    baseline_stats_path: str = "artifacts/baseline_stats.json"
    database_url: str = "sqlite:///./data/predictions.db"
    drift_window_size: int = Field(default=100, ge=10, le=10000)
    drift_psi_threshold: float = Field(default=0.25, gt=0.0, le=5.0)
    max_seq_len: int = Field(default=48, ge=8, le=512)
    ort_intra_op_threads: int = Field(default=2, ge=1, le=32)
    ort_inter_op_threads: int = Field(default=1, ge=1, le=16)
    warmup_runs: int = Field(default=3, ge=0, le=20)
    port: int = Field(default=8000, ge=1, le=65535)

    @property
    def model_file(self) -> Path:
        return Path(self.model_path)

    @property
    def tokenizer_file(self) -> Path:
        return Path(self.tokenizer_path)

    @property
    def baseline_stats_file(self) -> Path:
        return Path(self.baseline_stats_path)


def get_settings() -> Settings:
    return Settings()
