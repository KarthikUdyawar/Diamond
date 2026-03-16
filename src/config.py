"""
src/config.py
─────────────
Centralised configuration for the Diamond project.

Environment is selected at process startup by passing the correct env file:
    Local dev:  uv run --env-file .env.local  -m src.features
    Testing:    uv run --env-file .env.testing -m pytest
    Staging:    docker compose (env_file: .env.staging in docker-compose.yml)

All settings are validated by Pydantic on first access.
Call get_settings() anywhere — the result is cached for the process lifetime.
In tests, call get_settings.cache_clear() before patching env vars.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for every configurable value in the project.

    Pydantic reads values from environment variables (case-insensitive).
    No value is hardcoded here — all defaults are safe non-secret fallbacks
    suitable for local development only.
    """

    model_config = SettingsConfigDict(
        # Do NOT specify env_file here — the caller passes --env-file to uv
        # or sets env_file in docker-compose.yml. This keeps switching
        # explicit and avoids silent file-precedence surprises.
        protected_namespaces=("settings_",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unknown vars in .env.* files are silently ignored
    )

    # ── Environment ────────────────────────────────────────────────────────────
    app_env: Literal["local", "testing", "staging"] = Field(
        default="local",
        description="Active environment. Controls log renderer and strictness.",
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
        description=("Minimum log level. DEBUG for local, INFO for staging, WARNING for testing."),
    )

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description=(
            "MLflow tracking server URI. Use service name in Docker Compose (http://mlflow:5000)."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="diamond-price",
        description="MLflow experiment name. Shared across environments.",
    )

    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    postgres_user: str = Field(default="diamond")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="diamond")
    postgres_host: str = Field(
        default="localhost",
        description="Use service name in Docker Compose (postgres).",
    )
    postgres_port: int = Field(default=5432)

    # ── Kaggle ─────────────────────────────────────────────────────────────────
    kaggle_username: str = Field(default="")
    kaggle_key: str = Field(default="")

    # ── Paths ──────────────────────────────────────────────────────────────────
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory containing raw per-shape CSVs from Kaggle.",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for train.parquet, test.parquet, pipeline.joblib.",
    )

    # ── SHAP ───────────────────────────────────────────────────────────────────
    shap_background_rows: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Number of background rows for TreeExplainer.",
    )
    shap_positive_color: str = Field(
        default="#2a9d8f",
        description="Teal — colour for positive SHAP contributions.",
    )
    shap_negative_color: str = Field(
        default="#e76f51",
        description="Coral — colour for negative SHAP contributions.",
    )
    shap_random_state: int = Field(
        default=42,
        description="Random state for background sample selection.",
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model_registry_name: str = Field(
        default="Diamond",
        description="Name in MLflow Model Registry.",
    )
    model_confidence_pct: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="±% heuristic applied to predicted price for confidence range.",
    )
    optuna_n_trials: int = Field(
        default=50,
        ge=1,
        description="Number of Optuna trials for CatBoost hyperparameter search.",
    )
    train_test_split_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)

    # ── Validators ─────────────────────────────────────────────────────────────
    @field_validator("postgres_password", "kaggle_username", "kaggle_key", mode="before")
    @classmethod
    def _warn_empty_secrets(cls, v: str, info: object) -> str:
        """
        Warn (not fail) if a secret is empty in staging.
        Uses print() because the logger is not yet initialised at Settings
        construction time.
        """
        if not v and os.environ.get("APP_ENV", "local") == "staging":
            field_name = getattr(info, "field_name", "unknown")
            print(  # noqa: T201
                f"[config] WARNING: '{field_name}' is empty in staging environment."
            )
        return v

    # ── Derived properties ─────────────────────────────────────────────────────
    @property
    def postgres_dsn(self) -> str:
        """Full PostgreSQL DSN for SQLAlchemy / psycopg2."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def mlflow_model_uri(self) -> str:
        """URI for loading the production model from the registry."""
        return f"models:/{self.model_registry_name}/Production"

    @property
    def pipeline_path(self) -> Path:
        """Absolute path to the fitted pipeline joblib."""
        return self.processed_dir / "pipeline.joblib"

    @property
    def train_parquet_path(self) -> Path:
        return self.processed_dir / "train.parquet"

    @property
    def test_parquet_path(self) -> Path:
        return self.processed_dir / "test.parquet"

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"

    @property
    def is_staging(self) -> bool:
        return self.app_env == "staging"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.

    The cache is process-scoped — one Settings object per process lifetime.
    In tests, call get_settings.cache_clear() before patching env vars.

    Example
    -------
    >>> from src.config import get_settings
    >>> cfg = get_settings()
    >>> cfg.mlflow_tracking_uri
    'http://localhost:5000'
    """
    return Settings()
