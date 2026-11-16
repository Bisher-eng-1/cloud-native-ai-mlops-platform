from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, desc, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )
    text_length: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_preview: Mapped[str] = mapped_column(String(200), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_probability: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    request_source: Mapped[str] = mapped_column(Text, default="api", nullable=False)


def normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://") and "+psycopg" not in raw_url:
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url


def create_db_engine(database_url: str):
    normalized = normalize_database_url(database_url)
    connect_args = {}
    if normalized.startswith("sqlite"):
        if ":memory:" not in normalized:
            db_path = normalized.replace("sqlite:///", "", 1)
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False}
    return create_engine(normalized, pool_pre_ping=True, connect_args=connect_args)


class DatabaseManager:
    def __init__(self, database_url: str) -> None:
        self.database_url = normalize_database_url(database_url)
        self.engine = create_db_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def init_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def session(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def record_prediction(
        self,
        *,
        text_input: str,
        token_count: int,
        predicted_label: str,
        confidence: float,
        anomaly_probability: float,
        latency_ms: float,
        model_version: str,
        request_source: str = "api",
    ) -> PredictionLog:
        entry = PredictionLog(
            text_length=len(text_input),
            token_count=token_count,
            text_preview=text_input[:200],
            predicted_label=predicted_label,
            confidence=confidence,
            anomaly_probability=anomaly_probability,
            latency_ms=latency_ms,
            model_version=model_version,
            request_source=request_source,
        )
        with self.SessionLocal() as db:
            db.add(entry)
            db.commit()
            return entry

    def record_predictions_batch(self, items: list[dict]) -> list[PredictionLog]:
        entries = [
            PredictionLog(
                text_length=len(item["text_input"]),
                token_count=item["token_count"],
                text_preview=item["text_input"][:200],
                predicted_label=item["predicted_label"],
                confidence=item["confidence"],
                anomaly_probability=item["anomaly_probability"],
                latency_ms=item["latency_ms"],
                model_version=item["model_version"],
                request_source=item.get("request_source", "api_batch"),
            )
            for item in items
        ]
        with self.SessionLocal() as db:
            db.add_all(entries)
            db.commit()
            return entries

    def recent_predictions(self, limit: int = 100) -> list[PredictionLog]:
        with self.SessionLocal() as db:
            return (
                db.query(PredictionLog)
                .order_by(desc(PredictionLog.id))
                .limit(limit)
                .all()
            )

    def backend_kind(self) -> str:
        return "postgresql" if "postgresql" in self.database_url else "sqlite"

    def dispose(self) -> None:
        self.engine.dispose()
