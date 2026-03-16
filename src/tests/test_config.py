"""
src/tests/test_config.py
─────────────────────────
Unit tests for src/config.py.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestSettingsDefaults:
    def test_default_app_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APP_ENV", raising=False)
        assert Settings().app_env == "local"

    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        assert Settings().log_level == "DEBUG"

    def test_default_mlflow_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert Settings().mlflow_tracking_uri == "http://localhost:5000"

    def test_default_shap_background_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHAP_BACKGROUND_ROWS", raising=False)
        assert Settings().shap_background_rows == 200

    def test_default_confidence_pct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_CONFIDENCE_PCT", raising=False)
        assert Settings().model_confidence_pct == pytest.approx(0.08)

    def test_default_postgres_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        assert Settings().postgres_port == 5432


class TestSettingsEnvOverride:
    def test_app_env_staging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        assert Settings().app_env == "staging"

    def test_app_env_testing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        assert Settings().app_env == "testing"

    def test_log_level_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        assert Settings().log_level == "INFO"

    def test_mlflow_uri_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        assert Settings().mlflow_tracking_uri == "http://mlflow:5000"

    def test_shap_background_rows_testing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
        assert Settings().shap_background_rows == 10

    def test_optuna_trials_ci(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPTUNA_N_TRIALS", "2")
        assert Settings().optuna_n_trials == 2

    def test_postgres_host_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "postgres")
        assert Settings().postgres_host == "postgres"


class TestSettingsValidation:
    def test_invalid_app_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_log_level_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings()

    def test_shap_background_rows_below_min(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "5")
        with pytest.raises(ValidationError):
            Settings()

    def test_shap_background_rows_above_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "1001")
        with pytest.raises(ValidationError):
            Settings()

    def test_confidence_pct_above_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_CONFIDENCE_PCT", "1.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_split_size_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRAIN_TEST_SPLIT_SIZE", "0.0")
        with pytest.raises(ValidationError):
            Settings()

    def test_split_size_one_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRAIN_TEST_SPLIT_SIZE", "1.0")
        with pytest.raises(ValidationError):
            Settings()


class TestDerivedProperties:
    def test_postgres_dsn_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_USER", "diamond")
        monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
        monkeypatch.setenv("POSTGRES_HOST", "localhost")
        monkeypatch.setenv("POSTGRES_PORT", "5432")
        monkeypatch.setenv("POSTGRES_DB", "diamond")
        s = Settings()
        assert s.postgres_dsn == "postgresql://diamond:secret@localhost:5432/diamond"

    def test_mlflow_model_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_REGISTRY_NAME", "Diamond")
        assert Settings().mlflow_model_uri == "models:/Diamond/Production"

    def test_pipeline_path_derived(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROCESSED_DIR", "data/processed")
        s = Settings()
        assert s.pipeline_path.name == "pipeline.joblib"
        assert "processed" in str(s.pipeline_path)

    def test_is_local_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        s = Settings()
        assert s.is_local is True
        assert s.is_staging is False
        assert s.is_testing is False

    def test_is_staging_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        s = Settings()
        assert s.is_staging is True
        assert s.is_local is False

    def test_is_testing_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        s = Settings()
        assert s.is_testing is True


class TestGetSettingsCache:
    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_new_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        s1 = get_settings()
        get_settings.cache_clear()
        monkeypatch.setenv("APP_ENV", "testing")
        s2 = get_settings()
        assert s1 is not s2
        assert s2.app_env == "testing"
