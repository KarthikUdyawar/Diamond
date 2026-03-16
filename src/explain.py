"""
src/explain.py
──────────────
SHAP-based explainability for the Diamond price model.

This module provides two kinds of output:

  Global (per training run — logged to MLflow as artifacts):
    plot_summary()     → shap_summary.png        (beeswarm)
    plot_dependence()  → shap_dependence_weight.png
    plot_importance()  → shap_importance.png      (mean |SHAP| bar chart)

  Local (per prediction — served by the API):
    get_shap_values()    → dict[feature_name, shap_value]
    get_waterfall_b64()  → base64-encoded PNG waterfall chart

Design decisions:
    - ExplainerService is instantiated ONCE and cached at module level.
      get_explainer() returns the cached instance; never recreated per request.
    - Background sample: shap_background_rows rows, shap_random_state seed
      (both from config — defaults 200 and 42).
    - Colours: teal (#2a9d8f) positive, coral (#e76f51) negative.
    - All matplotlib figures are closed immediately after saving to prevent
      memory leaks in long-running API processes.
    - matplotlib backend set to "Agg" (non-interactive, safe in Docker).
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from src.config import get_settings
from src.logger import get_logger

# Use non-interactive backend — safe for headless Docker containers.
# Must be set before any other matplotlib import.
matplotlib.use("Agg")

logger = get_logger(__name__)

# ── Module-level explainer cache ───────────────────────────────────────────────
_explainer_cache: ExplainerService | None = None


class ExplainerService:
    """
    Wraps a shap.TreeExplainer with convenience methods for global and
    local SHAP outputs.

    Parameters
    ----------
    model : Any
        A fitted tree-based model (CatBoostRegressor, XGBRegressor, etc.)
        that shap.TreeExplainer supports natively.
    background_df : pd.DataFrame
        Feature matrix used to build the SHAP background distribution.
        Should be a random sample of the training data.
    feature_names : list[str]
        Ordered list of feature names matching the columns in background_df.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        background_df: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        cfg = get_settings()
        self._feature_names = feature_names
        self._pos_color = cfg.shap_positive_color
        self._neg_color = cfg.shap_negative_color

        logger.info(
            "explainer_init_started",
            background_rows=len(background_df),
            n_features=len(feature_names),
        )

        self._explainer = shap.TreeExplainer(
            model,
            data=background_df,
            feature_perturbation="interventional",
        )

        logger.info("explainer_init_done")

    # ── Local SHAP (per prediction) ────────────────────────────────────────────

    def get_shap_values(self, input_row: pd.DataFrame) -> dict[str, float]:
        """
        Compute SHAP values for a single input row.

        Parameters
        ----------
        input_row : pd.DataFrame
            Single-row DataFrame with the same columns as the training features.

        Returns
        -------
        dict[str, float]
            Mapping of feature name → SHAP value (in log-price space).
            Sorted descending by absolute value so the most influential
            feature is first.
        """
        logger.debug("shap_values_requested", shape=input_row.shape)

        raw = self._explainer.shap_values(input_row)
        values: np.ndarray = np.array(raw)
        if values.ndim == 2:  # shape (1, n_features) → (n_features,)
            values = values[0]

        result = dict(zip(self._feature_names, values.tolist(), strict=True))
        result = dict(sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True))

        logger.debug("shap_values_done", n_features=len(result))
        return result

    def get_waterfall_b64(self, input_row: pd.DataFrame) -> str:
        """
        Generate a SHAP waterfall chart for a single input row.

        Returns
        -------
        str
            Base64-encoded PNG string (UTF-8, no ``data:`` prefix).

        Notes
        -----
        Teal bars  (#2a9d8f) for positive SHAP contributions.
        Coral bars (#e76f51) for negative SHAP contributions.
        """
        logger.debug("waterfall_requested")

        explanation = self._explainer(input_row)
        exp = explanation[0] if hasattr(explanation, "__len__") else explanation

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, max_display=14, show=False)
        _recolor_bars(ax, self._pos_color, self._neg_color)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)

        encoded = base64.b64encode(buf.read()).decode("utf-8")
        logger.debug("waterfall_done", encoded_bytes=len(encoded))
        return encoded

    # ── Global SHAP (per training run) ─────────────────────────────────────────

    def plot_summary(
        self,
        X: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Generate a SHAP summary beeswarm plot.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (full training set or a representative sample).
        output_dir : Path, optional
            Directory to write the PNG. Defaults to a temporary directory.

        Returns
        -------
        Path
            Absolute path to ``shap_summary.png``.
        """
        logger.info("shap_summary_plot_started", rows=len(X))

        shap_values = self._explainer.shap_values(X)
        out_path = _resolve_output_path(output_dir, "shap_summary.png")

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self._feature_names,
            plot_type="dot",
            show=False,
            color_bar=True,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info("shap_summary_plot_done", path=str(out_path))
        return out_path

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str = "Weight",
        output_dir: Path | None = None,
    ) -> Path:
        """
        Generate a SHAP dependence plot for a single feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        feature : str
            Feature name to plot. Must be present in ``self._feature_names``.
        output_dir : Path, optional
            Directory to write the PNG. Defaults to a temporary directory.

        Returns
        -------
        Path
            Absolute path to ``shap_dependence_<feature>.png``.

        Raises
        ------
        ValueError
            If ``feature`` is not in the known feature list.
        """
        logger.info("shap_dependence_plot_started", feature=feature)

        if feature not in self._feature_names:
            msg = f"Feature '{feature}' not found in feature list. Available: {self._feature_names}"
            raise ValueError(msg)

        shap_values = self._explainer.shap_values(X)
        feature_idx = self._feature_names.index(feature)

        filename = f"shap_dependence_{feature.lower()}.png"
        out_path = _resolve_output_path(output_dir, filename)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=self._feature_names,
            ax=ax,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info("shap_dependence_plot_done", feature=feature, path=str(out_path))
        return out_path

    def plot_importance(
        self,
        X: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Generate a mean absolute SHAP bar chart (global feature importance).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        output_dir : Path, optional
            Directory to write the PNG. Defaults to a temporary directory.

        Returns
        -------
        Path
            Absolute path to ``shap_importance.png``.
        """
        logger.info("shap_importance_plot_started", rows=len(X))

        shap_values = self._explainer.shap_values(X)
        mean_abs: np.ndarray = np.abs(np.array(shap_values)).mean(axis=0)

        sorted_idx = np.argsort(mean_abs)[::-1]
        sorted_names = [self._feature_names[i] for i in sorted_idx]
        sorted_vals = mean_abs[sorted_idx]

        out_path = _resolve_output_path(output_dir, "shap_importance.png")

        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_names) * 0.45)))
        bars = ax.barh(sorted_names[::-1], sorted_vals[::-1])
        for bar in bars:
            bar.set_facecolor(self._pos_color)

        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.info("shap_importance_plot_done", path=str(out_path))
        return out_path


# ── Cache management ───────────────────────────────────────────────────────────


def init_explainer(
    model: Any,  # noqa: ANN401
    X_train: pd.DataFrame,
    feature_names: list[str],
) -> ExplainerService:
    """
    Initialise and cache the module-level ExplainerService.

    Call once after model training (in train.py) or once on API startup
    (in src/api/main.py lifespan). Subsequent get_explainer() calls return
    the cached instance without re-fitting.

    Parameters
    ----------
    model : Any
        Fitted CatBoostRegressor (or any tree model).
    X_train : pd.DataFrame
        Full training feature matrix. Background sample is drawn from here.
    feature_names : list[str]
        Ordered feature names matching columns in X_train.

    Returns
    -------
    ExplainerService
        The newly created (and now cached) explainer.
    """
    global _explainer_cache  # noqa: PLW0603

    cfg = get_settings()
    rng = np.random.default_rng(cfg.shap_random_state)
    n = min(cfg.shap_background_rows, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    background = X_train.iloc[idx].reset_index(drop=True)

    logger.info(
        "explainer_cache_initialising",
        background_rows=n,
        total_train_rows=len(X_train),
    )

    _explainer_cache = ExplainerService(
        model=model,
        background_df=background,
        feature_names=feature_names,
    )
    return _explainer_cache


def get_explainer() -> ExplainerService:
    """
    Return the cached ExplainerService.

    Raises
    ------
    RuntimeError
        If init_explainer() has not been called yet.
    """
    if _explainer_cache is None:
        msg = (
            "ExplainerService has not been initialised. "
            "Call init_explainer(model, X_train, feature_names) first."
        )
        raise RuntimeError(msg)
    return _explainer_cache


def clear_explainer() -> None:
    """
    Clear the module-level explainer cache.

    Used in tests to reset state between runs. Not needed in production.
    """
    global _explainer_cache  # noqa: PLW0603
    _explainer_cache = None
    logger.debug("explainer_cache_cleared")


# ── Private helpers ────────────────────────────────────────────────────────────


def _resolve_output_path(output_dir: Path | None, filename: str) -> Path:
    """Return a validated output path, creating the directory if needed."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _recolor_bars(ax: Axes, pos_color: str, neg_color: str) -> None:
    """
    Walk the patches on a rendered waterfall plot and apply project colours.

    SHAP's default waterfall uses blue/red. This function replaces those
    with teal (positive) and coral (negative) after shap renders to axes.
    """
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            width = patch.get_width()
            if width > 0:
                patch.set_facecolor(pos_color)
            elif width < 0:
                patch.set_facecolor(neg_color)
