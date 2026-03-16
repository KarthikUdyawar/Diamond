"""
src/tests/test_explain.py
──────────────────────────
Unit tests for src/explain.py.

shap.TreeExplainer is mocked in all tests — no real model or SHAP
computation happens, keeping the suite fast and dependency-free.
"""

from __future__ import annotations

import base64
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import get_settings
from src.explain import (
    ExplainerService,
    clear_explainer,
    get_explainer,
    init_explainer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state() -> Generator[None, None, None]:
    """Clear explainer cache and settings cache before/after each test."""
    clear_explainer()
    get_settings.cache_clear()
    yield
    clear_explainer()
    get_settings.cache_clear()


@pytest.fixture()
def feature_names() -> list[str]:
    return ["Weight", "Colour", "Clarity", "volume", "log_weight"]


@pytest.fixture()
def sample_df(feature_names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((20, len(feature_names))), columns=feature_names)


@pytest.fixture()
def mock_tree_explainer(feature_names: list[str]) -> MagicMock:
    """Mock shap.TreeExplainer returning deterministic SHAP values."""
    n = len(feature_names)
    mock = MagicMock()
    # shap_values called on full matrix → shape (n_rows, n_features)
    mock.shap_values.return_value = np.ones((20, n)) * 0.1
    return mock


@pytest.fixture()
def mock_model() -> MagicMock:
    m = MagicMock()
    m.predict.return_value = np.ones(20)
    return m


@pytest.fixture()
def explainer(
    mock_model: MagicMock,
    sample_df: pd.DataFrame,
    feature_names: list[str],
    mock_tree_explainer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> ExplainerService:
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
    monkeypatch.setenv("SHAP_POSITIVE_COLOR", "#2a9d8f")
    monkeypatch.setenv("SHAP_NEGATIVE_COLOR", "#e76f51")
    get_settings.cache_clear()  # ensure the env vars above are picked up
    with patch("src.explain.shap.TreeExplainer", return_value=mock_tree_explainer):
        svc = ExplainerService(
            model=mock_model,
            background_df=sample_df,
            feature_names=feature_names,
        )
    return svc


def mock_shap_values(explainer: ExplainerService, values: np.ndarray) -> None:
    explainer._explainer.shap_values = MagicMock(return_value=values)


# ---------------------------------------------------------------------------
# ExplainerService init
# ---------------------------------------------------------------------------


class TestExplainerServiceInit:
    def test_feature_names_stored(
        self, explainer: ExplainerService, feature_names: list[str]
    ) -> None:
        assert explainer._feature_names == feature_names

    def test_colors_from_config(self, explainer: ExplainerService) -> None:
        assert explainer._pos_color == "#2a9d8f"
        assert explainer._neg_color == "#e76f51"

    def test_custom_colors_via_env(
        self,
        mock_model: MagicMock,
        sample_df: pd.DataFrame,
        feature_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
        monkeypatch.setenv("SHAP_POSITIVE_COLOR", "#ffffff")
        monkeypatch.setenv("SHAP_NEGATIVE_COLOR", "#000000")
        get_settings.cache_clear()
        with patch("src.explain.shap.TreeExplainer", return_value=MagicMock()):
            svc = ExplainerService(mock_model, sample_df, feature_names)
        assert svc._pos_color == "#ffffff"
        assert svc._neg_color == "#000000"


# ---------------------------------------------------------------------------
# get_shap_values
# ---------------------------------------------------------------------------


class TestGetShapValues:
    def test_returns_dict(
        self,
        explainer: ExplainerService,
        sample_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        n = len(feature_names)
        vals = np.array([[0.1] * n])
        mock_shap_values(explainer, vals)
        result = explainer.get_shap_values(sample_df.iloc[[0]])
        assert isinstance(result, dict)

    def test_keys_match_feature_names(
        self,
        explainer: ExplainerService,
        sample_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        n = len(feature_names)
        vals = np.array([[float(i) for i in range(n)]])
        mock_shap_values(explainer, vals)
        result = explainer.get_shap_values(sample_df.iloc[[0]])
        assert set(result.keys()) == set(feature_names)

    def test_sorted_by_absolute_value_descending(
        self,
        explainer: ExplainerService,
        sample_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        vals = np.array([[0.1, -0.5, 0.3, 0.01, -0.4]])
        mock_shap_values(explainer, vals)
        result = explainer.get_shap_values(sample_df.iloc[[0]])
        abs_vals = [abs(v) for v in result.values()]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_values_are_floats(
        self,
        explainer: ExplainerService,
        sample_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        n = len(feature_names)
        vals = np.ones((1, n)) * 0.5
        mock_shap_values(explainer, vals)
        result = explainer.get_shap_values(sample_df.iloc[[0]])
        for v in result.values():
            assert isinstance(v, float)

    def test_handles_2d_array_output(
        self,
        explainer: ExplainerService,
        sample_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        """shap_values can return (1, n_features) — must be flattened."""
        n = len(feature_names)
        vals = np.ones((1, n)) * 0.2
        mock_shap_values(explainer, vals)
        result = explainer.get_shap_values(sample_df.iloc[[0]])
        assert len(result) == n


# ---------------------------------------------------------------------------
# get_waterfall_b64
# ---------------------------------------------------------------------------


class TestGetWaterfallB64:
    def test_returns_non_empty_string(
        self, explainer: ExplainerService, sample_df: pd.DataFrame
    ) -> None:
        mock_exp = MagicMock()
        mock_exp.__getitem__ = MagicMock(return_value=mock_exp)
        mock_exp.__len__ = MagicMock(return_value=1)
        mock_shap_values(explainer, mock_exp)

        with (
            patch("src.explain.shap.waterfall_plot"),
            patch("src.explain._recolor_bars"),
        ):
            result = explainer.get_waterfall_b64(sample_df.iloc[[0]])

        assert isinstance(result, str)
        assert len(result) > 0

    def test_output_is_valid_base64(
        self, explainer: ExplainerService, sample_df: pd.DataFrame
    ) -> None:
        mock_exp = MagicMock()
        mock_exp.__getitem__ = MagicMock(return_value=mock_exp)
        mock_exp.__len__ = MagicMock(return_value=1)
        mock_shap_values(explainer, mock_exp)

        with (
            patch("src.explain.shap.waterfall_plot"),
            patch("src.explain._recolor_bars"),
        ):
            result = explainer.get_waterfall_b64(sample_df.iloc[[0]])

        decoded = base64.b64decode(result)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# Global plots
# ---------------------------------------------------------------------------


class TestGlobalPlots:
    def test_plot_summary_creates_png(
        self, explainer: ExplainerService, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        with patch("src.explain.shap.summary_plot"):
            result = explainer.plot_summary(sample_df, output_dir=tmp_path)
        assert result.exists()
        assert result.name == "shap_summary.png"

    def test_plot_dependence_creates_png(
        self, explainer: ExplainerService, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        with patch("src.explain.shap.dependence_plot"):
            result = explainer.plot_dependence(sample_df, feature="Weight", output_dir=tmp_path)
        assert result.exists()
        assert "weight" in result.name

    def test_plot_dependence_unknown_feature_raises(
        self, explainer: ExplainerService, sample_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            explainer.plot_dependence(sample_df, feature="NonExistent")

    def test_plot_importance_creates_png(
        self, explainer: ExplainerService, sample_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        result = explainer.plot_importance(sample_df, output_dir=tmp_path)
        assert result.exists()
        assert result.name == "shap_importance.png"

    def test_plot_summary_uses_temp_dir_by_default(
        self, explainer: ExplainerService, sample_df: pd.DataFrame
    ) -> None:
        with patch("src.explain.shap.summary_plot"):
            result = explainer.plot_summary(sample_df, output_dir=None)
        assert result.exists()


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


class TestExplainerCache:
    def test_get_explainer_raises_before_init(self) -> None:
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_explainer()

    def test_init_sets_cache(
        self,
        mock_model: MagicMock,
        sample_df: pd.DataFrame,
        feature_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
        with patch("src.explain.shap.TreeExplainer", return_value=MagicMock()):
            svc = init_explainer(mock_model, sample_df, feature_names)
        assert get_explainer() is svc

    def test_get_explainer_same_instance(
        self,
        mock_model: MagicMock,
        sample_df: pd.DataFrame,
        feature_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
        with patch("src.explain.shap.TreeExplainer", return_value=MagicMock()):
            svc = init_explainer(mock_model, sample_df, feature_names)
        assert get_explainer() is svc
        assert get_explainer() is svc  # second call — same instance

    def test_clear_resets_cache(
        self,
        mock_model: MagicMock,
        sample_df: pd.DataFrame,
        feature_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "10")
        with patch("src.explain.shap.TreeExplainer", return_value=MagicMock()):
            init_explainer(mock_model, sample_df, feature_names)
        clear_explainer()
        with pytest.raises(RuntimeError):
            get_explainer()

    def test_init_samples_background_rows(
        self,
        mock_model: MagicMock,
        sample_df: pd.DataFrame,
        feature_names: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Use 15 — valid (ge=10), and less than the 20-row sample_df,
        # so we can verify the sampling cap is applied correctly.
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("SHAP_BACKGROUND_ROWS", "15")
        get_settings.cache_clear()

        captured_bg: list[pd.DataFrame] = []

        def capture_init(model: object, data: pd.DataFrame, **kw: object) -> MagicMock:
            captured_bg.append(data)
            return MagicMock()

        with patch("src.explain.shap.TreeExplainer", side_effect=capture_init):
            init_explainer(mock_model, sample_df, feature_names)

        assert len(captured_bg[0]) == 15
