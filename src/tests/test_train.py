"""
src/tests/test_train.py
Unit tests for src/train.py.

All MLflow, model, and filesystem interactions are mocked — no live server
or real training required. Tests focus on:
  - Metric computation correctness
  - Feature name derivation
  - Skip-if-exists logic
  - _register_best_model calls the right client methods
  - Orchestrator wires all pieces together correctly
  - pipeline_path and processed_dir propagate end-to-end (regression for
    the bug where internal callers fell back to src.constants.PIPELINE_PATH)
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        from src.train import _compute_metrics

        y = np.array([1.0, 2.0, 3.0])
        m = _compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)
        assert m["r2"] == pytest.approx(1.0)
        assert m["mape"] == pytest.approx(0.0, abs=1e-6)

    def test_known_rmse(self) -> None:
        from src.train import _compute_metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        m = _compute_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(1.0)

    def test_returns_four_keys(self) -> None:
        from src.train import _compute_metrics

        m = _compute_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        assert set(m.keys()) == {"rmse", "mae", "r2", "mape"}

    def test_mape_excludes_near_zero_actuals(self) -> None:
        from src.train import _compute_metrics

        # near-zero actual should be excluded, not cause division by zero
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([0.1, 2.0, 4.0])
        m = _compute_metrics(y_true, y_pred)
        assert np.isfinite(m["mape"])

    def test_r2_below_zero_for_bad_model(self) -> None:
        from src.train import _compute_metrics

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        m = _compute_metrics(y_true, y_pred)
        assert m["r2"] < 0


# ---------------------------------------------------------------------------
# Feature name derivation
# ---------------------------------------------------------------------------


class TestGetFeatureNames:
    def _make_preprocessor(self) -> MagicMock:
        """Build a mock ColumnTransformer matching the real pipeline layout."""
        ohe = MagicMock()
        ohe.categories_ = [["Cushion", "Round", "Oval"]]

        ordinal_tf = MagicMock()
        ordinal_tf.named_steps = {"encoder": MagicMock()}

        nominal_tf = MagicMock()
        nominal_tf.named_steps = {"encoder": ohe}

        numeric_tf = MagicMock()

        preprocessor = MagicMock()
        preprocessor.transformers_ = [
            ("ordinal", ordinal_tf, ["Cut", "Colour"]),
            ("nominal", nominal_tf, ["Shape"]),
            ("numeric", numeric_tf, ["Weight", "length"]),
        ]
        return preprocessor

    def test_feature_names_order(self) -> None:
        from src.train import _get_feature_names

        prep = self._make_preprocessor()
        names = _get_feature_names(prep)
        assert names == [
            "Cut",
            "Colour",
            "Shape_Cushion",
            "Shape_Round",
            "Shape_Oval",
            "Weight",
            "length",
        ]

    def test_ohe_prefix(self) -> None:
        from src.train import _get_feature_names

        prep = self._make_preprocessor()
        names = _get_feature_names(prep)
        ohe_names = [n for n in names if n.startswith("Shape_")]
        assert ohe_names == ["Shape_Cushion", "Shape_Round", "Shape_Oval"]

    def test_total_count(self) -> None:
        from src.train import _get_feature_names

        prep = self._make_preprocessor()
        names = _get_feature_names(prep)
        # 2 ordinal + 3 OHE + 2 numeric = 7
        assert len(names) == 7


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    def test_returns_correct_count(self) -> None:
        from src.train import _select_features

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = rng.standard_normal(100)
        feature_names = [f"f{i}" for i in range(10)]

        with patch("src.train.CatBoostRegressor") as MockCB:
            instance = MockCB.return_value
            instance.get_feature_importance.return_value = np.arange(10, dtype=float)
            indices, names = _select_features(X, y, feature_names, n_features=5)

        assert len(indices) == 5
        assert len(names) == 5

    def test_indices_sorted(self) -> None:
        from src.train import _select_features

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)
        feature_names = [f"feat_{i}" for i in range(8)]

        with patch("src.train.CatBoostRegressor") as MockCB:
            instance = MockCB.return_value
            instance.get_feature_importance.return_value = np.arange(8, dtype=float)
            indices, _ = _select_features(X, y, feature_names, n_features=3)

        assert indices == sorted(indices)

    def test_names_match_indices(self) -> None:
        from src.train import _select_features

        feature_names = ["a", "b", "c", "d", "e"]
        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 5))
        y = rng.standard_normal(30)

        with patch("src.train.CatBoostRegressor") as MockCB:
            instance = MockCB.return_value
            instance.get_feature_importance.return_value = np.array(
                [5.0, 4.0, 3.0, 2.0, 1.0]
            )
            indices, names = _select_features(X, y, feature_names, n_features=3)

        for idx, name in zip(indices, names, strict=True):
            assert feature_names[idx] == name


# ---------------------------------------------------------------------------
# Skip-if-exists logic
# ---------------------------------------------------------------------------


class TestRunExists:
    def test_returns_true_when_run_found(self) -> None:
        from src.train import _run_exists

        mock_run = MagicMock()
        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.search_runs.return_value = [mock_run]
            assert _run_exists("exp_123", "catboost-baseline") is True

    def test_returns_false_when_no_run(self) -> None:
        from src.train import _run_exists

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.search_runs.return_value = []
            assert _run_exists("exp_123", "catboost-baseline") is False

    def test_search_uses_run_name_filter(self) -> None:
        from src.train import _run_exists

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.search_runs.return_value = []
            _run_exists("exp_1", "my-run")
            call_kwargs = mock_client.search_runs.call_args
            assert "my-run" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------


class TestGetOrCreateExperiment:
    def test_creates_when_missing(self) -> None:
        from src.train import _get_or_create_experiment

        with (
            patch("src.train.mlflow.get_experiment_by_name", return_value=None),
            patch("src.train.mlflow.MlflowClient"),
            patch(
                "src.train.mlflow.create_experiment", return_value="42"
            ) as mock_create,
        ):
            result = _get_or_create_experiment("new-experiment")
            mock_create.assert_called_once_with("new-experiment")
            assert result == "42"

    def test_returns_existing_id(self) -> None:
        from src.train import _get_or_create_experiment

        mock_exp = MagicMock()
        mock_exp.experiment_id = "99"
        mock_exp.lifecycle_stage = "active"
        with (
            patch("src.train.mlflow.get_experiment_by_name", return_value=mock_exp),
            patch("src.train.mlflow.MlflowClient"),
        ):
            result = _get_or_create_experiment("existing")
            assert result == "99"

    def test_restores_deleted_experiment(self) -> None:
        from src.train import _get_or_create_experiment

        deleted_exp = MagicMock()
        deleted_exp.experiment_id = "1"
        deleted_exp.lifecycle_stage = "deleted"

        with (
            patch("src.train.mlflow.get_experiment_by_name", return_value=deleted_exp),
            patch("src.train.mlflow.MlflowClient") as MockClient,
        ):
            result = _get_or_create_experiment("diamond-price")
            MockClient.return_value.restore_experiment.assert_called_once_with("1")
            assert result == "1"

    def test_returns_active_experiment_without_touching_it(self) -> None:
        from src.train import _get_or_create_experiment

        active_exp = MagicMock()
        active_exp.experiment_id = "2"
        active_exp.lifecycle_stage = "active"

        with (
            patch("src.train.mlflow.get_experiment_by_name", return_value=active_exp),
            patch("src.train.mlflow.MlflowClient") as MockClient,
        ):
            result = _get_or_create_experiment("diamond-price")
            MockClient.return_value.restore_experiment.assert_not_called()
            assert result == "2"


# ---------------------------------------------------------------------------
# _train_and_log
# ---------------------------------------------------------------------------


class TestTrainAndLog:
    def _make_mock_model(self) -> MagicMock:
        model = MagicMock()
        model.get_params.return_value = {"iterations": 100, "lr": 0.05}
        model.predict.return_value = np.array([1.0, 2.0, 3.0])
        return model

    def test_returns_run_id_and_rmse(self) -> None:
        from src.train import _train_and_log

        mock_model = self._make_mock_model()
        y = np.array([1.0, 2.0, 3.0])

        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"

        with (
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch("src.train.Path.exists", return_value=False),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            run_id, rmse = _train_and_log(
                mock_model,
                "catboost",
                "test-run",
                "exp_1",
                np.ones((3, 4)),
                np.ones((3, 4)),
                y,
                y,
                ["f1", "f2", "f3", "f4"],
            )

        assert run_id == "abc123"
        assert isinstance(rmse, float)

    def test_logs_training_time(self) -> None:
        from src.train import _train_and_log

        mock_model = self._make_mock_model()
        y = np.array([1.0, 2.0, 3.0])
        mock_run = MagicMock()
        mock_run.info.run_id = "xyz"

        logged_metrics: dict[str, float] = {}

        def capture_metrics(m: dict[str, float]) -> None:
            logged_metrics.update(m)

        with (
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics", side_effect=capture_metrics),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch("src.train.Path.exists", return_value=False),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)
            _train_and_log(
                mock_model,
                "catboost",
                "test-run",
                "exp_1",
                np.ones((3, 4)),
                np.ones((3, 4)),
                y,
                y,
                ["f1", "f2", "f3", "f4"],
            )

        assert "training_time_sec" in logged_metrics

    def test_custom_pipeline_path_used_for_artifact(self) -> None:
        """pipeline_path argument must be used — not the global PIPELINE_PATH."""
        from src.train import _train_and_log

        mock_model = self._make_mock_model()
        y = np.array([1.0, 2.0, 3.0])
        mock_run = MagicMock()
        mock_run.info.run_id = "r1"

        artifact_calls: list[str] = []

        with (
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch(
                "src.train.mlflow.log_artifact",
                side_effect=lambda *a, **kw: artifact_calls.append(a[0]),
            ),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch("src.train.Path.exists", return_value=True),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)
            _train_and_log(
                mock_model,
                "catboost",
                "run",
                "exp",
                np.ones((3, 2)),
                np.ones((3, 2)),
                y,
                y,
                ["a", "b"],
                pipeline_path="/custom/path/pipeline.joblib",
            )

        assert "/custom/path/pipeline.joblib" in artifact_calls


# ---------------------------------------------------------------------------
# _register_best_model
# ---------------------------------------------------------------------------


class TestRegisterBestModel:
    def test_registers_and_promotes(self) -> None:
        from src.train import _register_best_model

        mock_version = MagicMock()
        mock_version.version = "1"

        mock_client_instance = MagicMock()

        with (
            patch(
                "src.train.mlflow.register_model", return_value=mock_version
            ) as mock_reg,
            patch("src.train.mlflow.MlflowClient", return_value=mock_client_instance),
            patch("src.train._verify_model_loads"),
        ):
            _register_best_model("run_abc", model_name="Diamond", stage="Production")

            mock_reg.assert_called_once_with(
                model_uri="runs:/run_abc/model", name="Diamond"
            )
            mock_client_instance.transition_model_version_stage.assert_called_once_with(
                name="Diamond",
                version="1",
                stage="Production",
                archive_existing_versions=True,
            )

    def test_verifies_load(self) -> None:
        from src.train import _register_best_model

        mock_version = MagicMock()
        mock_version.version = "2"

        with (
            patch("src.train.mlflow.register_model", return_value=mock_version),
            patch("src.train.mlflow.MlflowClient"),
            patch("src.train._verify_model_loads") as mock_verify,
        ):
            _register_best_model("run_xyz")
            mock_verify.assert_called_once_with("Diamond", "Production")


# ---------------------------------------------------------------------------
# _is_already_registered
# ---------------------------------------------------------------------------


class TestIsAlreadyRegistered:
    def test_returns_true_when_staged_version_matches_run_id(self) -> None:
        from src.train import _is_already_registered

        mv = MagicMock()
        mv.run_id = "abc123"

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.get_latest_versions.return_value = [mv]
            assert _is_already_registered("abc123") is True

    def test_returns_false_when_staged_version_differs(self) -> None:
        from src.train import _is_already_registered

        mv = MagicMock()
        mv.run_id = "different_run"

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.get_latest_versions.return_value = [mv]
            assert _is_already_registered("abc123") is False

    def test_returns_false_when_no_staged_versions(self) -> None:
        from src.train import _is_already_registered

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.get_latest_versions.return_value = []
            assert _is_already_registered("abc123") is False

    def test_returns_false_when_model_does_not_exist(self) -> None:
        """get_latest_versions raises if model name unknown — must return False."""
        from src.train import _is_already_registered

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.get_latest_versions.side_effect = Exception(
                "RESOURCE_DOES_NOT_EXIST"
            )
            assert _is_already_registered("abc123") is False

    def test_queries_correct_stage(self) -> None:
        from src.train import _is_already_registered

        with patch("src.train.mlflow.MlflowClient") as MockClient:
            MockClient.return_value.get_latest_versions.return_value = []
            _is_already_registered("abc123", model_name="Diamond", stage="Staging")
            MockClient.return_value.get_latest_versions.assert_called_once_with(
                "Diamond", stages=["Staging"]
            )


# ---------------------------------------------------------------------------
# Orchestrator — run_training
# ---------------------------------------------------------------------------


class TestRunTraining:
    def _make_load_data_return(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 22))
        y = rng.standard_normal(100)
        feature_names = [f"feat_{i}" for i in range(22)]
        return X, X, y, y, feature_names

    def test_skips_existing_runs(self) -> None:
        """If all runs already exist and the staged model already points to the
        best run_id, no trainer function and no registration should be called."""
        from src.train import run_training

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch("src.train._load_data", return_value=self._make_load_data_return()),
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=True),
            patch("src.train.mlflow.MlflowClient") as MockClient,
            patch("src.train._register_best_model") as mock_register,
            patch("src.train._is_already_registered", return_value=True),
            patch("src.train._verify_model_loads"),
        ):
            mock_run = MagicMock()
            mock_run.info.run_id = "existing_run"
            mock_run.data.metrics = {"rmse": 0.5}
            MockClient.return_value.search_runs.return_value = [mock_run]

            with (
                patch("src.train._train_catboost_baseline") as mock_cb,
                patch("src.train._train_xgboost_baseline") as mock_xgb,
                patch("src.train._train_lightgbm_baseline") as mock_lgbm,
                patch("src.train._train_gbm_baseline") as mock_gbm,
                patch("src.train._run_optuna_tuning") as mock_optuna,
            ):
                run_training()
                mock_cb.assert_not_called()
                mock_xgb.assert_not_called()
                mock_lgbm.assert_not_called()
                mock_gbm.assert_not_called()
                mock_optuna.assert_not_called()
                # Already registered — must not create a duplicate version
                mock_register.assert_not_called()

    def test_trains_all_when_no_existing_runs(self) -> None:
        """If no runs exist, all 4 baselines + optuna should be called."""
        from src.train import run_training

        dummy_result = ("run_id_x", 0.42)

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch("src.train._load_data", return_value=self._make_load_data_return()),
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._register_best_model"),
            patch("src.train._is_already_registered", return_value=False),
            patch(
                "src.train._train_catboost_baseline", return_value=dummy_result
            ) as mock_cb,
            patch(
                "src.train._train_xgboost_baseline", return_value=dummy_result
            ) as mock_xgb,
            patch(
                "src.train._train_lightgbm_baseline", return_value=dummy_result
            ) as mock_lgbm,
            patch(
                "src.train._train_gbm_baseline", return_value=dummy_result
            ) as mock_gbm,
            patch(
                "src.train._run_optuna_tuning", return_value=dummy_result
            ) as mock_optuna,
        ):
            run_training()
            mock_cb.assert_called_once()
            mock_xgb.assert_called_once()
            mock_lgbm.assert_called_once()
            mock_gbm.assert_called_once()
            mock_optuna.assert_called_once()

    def test_registers_best_run(self) -> None:
        """Best run (lowest RMSE) should be passed to _register_best_model."""
        from src.train import run_training

        results = {
            "catboost-baseline": ("run_cb", 0.50),
            "xgboost-baseline": ("run_xgb", 0.45),
            "lightgbm-baseline": ("run_lgbm", 0.55),
            "gbm-baseline": ("run_gbm", 0.60),
            "catboost-tuned": ("run_tuned", 0.30),  # ← best
        }

        call_sequence = list(results.values())

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch("src.train._load_data", return_value=self._make_load_data_return()),
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._train_catboost_baseline", return_value=call_sequence[0]),
            patch("src.train._train_xgboost_baseline", return_value=call_sequence[1]),
            patch("src.train._train_lightgbm_baseline", return_value=call_sequence[2]),
            patch("src.train._train_gbm_baseline", return_value=call_sequence[3]),
            patch("src.train._run_optuna_tuning", return_value=call_sequence[4]),
            patch("src.train._is_already_registered", return_value=False),
            patch("src.train._register_best_model") as mock_register,
        ):
            run_training()
            mock_register.assert_called_once_with("run_tuned")

    def test_pipeline_path_forwarded_to_load_data(self) -> None:
        """run_training(pipeline_path=X) must reach _load_data(pipeline_path=X)."""
        from src.train import run_training

        dummy = ("rid", 0.4)

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch(
                "src.train._load_data", return_value=self._make_load_data_return()
            ) as mock_load,
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._train_catboost_baseline", return_value=dummy),
            patch("src.train._train_xgboost_baseline", return_value=dummy),
            patch("src.train._train_lightgbm_baseline", return_value=dummy),
            patch("src.train._train_gbm_baseline", return_value=dummy),
            patch("src.train._run_optuna_tuning", return_value=dummy),
            patch("src.train._is_already_registered", return_value=False),
            patch("src.train._register_best_model"),
        ):
            run_training(pipeline_path="/custom/pipeline.joblib")
            _, kwargs = mock_load.call_args
            assert kwargs.get("pipeline_path") == "/custom/pipeline.joblib"

    def test_processed_dir_forwarded_to_load_data(self) -> None:
        """run_training(processed_dir=X) must reach _load_data(processed_dir=X)."""
        from src.train import run_training

        dummy = ("rid", 0.4)

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch(
                "src.train._load_data", return_value=self._make_load_data_return()
            ) as mock_load,
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._train_catboost_baseline", return_value=dummy),
            patch("src.train._train_xgboost_baseline", return_value=dummy),
            patch("src.train._train_lightgbm_baseline", return_value=dummy),
            patch("src.train._train_gbm_baseline", return_value=dummy),
            patch("src.train._run_optuna_tuning", return_value=dummy),
            patch("src.train._is_already_registered", return_value=False),
            patch("src.train._register_best_model"),
        ):
            run_training(processed_dir="/custom/processed")
            _, kwargs = mock_load.call_args
            assert kwargs.get("processed_dir") == "/custom/processed"

    def test_pipeline_path_forwarded_to_baseline_trainers(self) -> None:
        """Custom pipeline_path must reach each baseline wrapper,
        not fall back to constant."""
        from src.train import run_training

        dummy = ("rid", 0.4)
        captured: list[str] = []

        def capture_pipeline(*args: object, **kwargs: object) -> tuple[str, float]:
            captured.append(str(kwargs.get("pipeline_path", "")))
            return dummy

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch("src.train._load_data", return_value=self._make_load_data_return()),
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._train_catboost_baseline", side_effect=capture_pipeline),
            patch("src.train._train_xgboost_baseline", side_effect=capture_pipeline),
            patch("src.train._train_lightgbm_baseline", side_effect=capture_pipeline),
            patch("src.train._train_gbm_baseline", side_effect=capture_pipeline),
            patch("src.train._run_optuna_tuning", return_value=dummy),
            patch("src.train._is_already_registered", return_value=False),
            patch("src.train._register_best_model"),
        ):
            run_training(pipeline_path="/my/pipeline.joblib")

        assert len(captured) == 4
        assert all(p == "/my/pipeline.joblib" for p in captured)

    def test_pipeline_path_forwarded_to_optuna(self) -> None:
        """Custom pipeline_path must reach _run_optuna_tuning,
        not fall back to constant."""
        from src.train import run_training

        dummy = ("rid", 0.4)
        captured: list[str] = []

        def capture_optuna(*args: object, **kwargs: object) -> tuple[str, float]:
            captured.append(str(kwargs.get("pipeline_path", "")))
            return dummy

        with (
            patch("src.train.mlflow.set_tracking_uri"),
            patch("src.train._get_or_create_experiment", return_value="exp_1"),
            patch("src.train._load_data", return_value=self._make_load_data_return()),
            patch(
                "src.train._select_features",
                return_value=([0, 1, 2, 3], ["f0", "f1", "f2", "f3"]),
            ),
            patch("src.train._run_exists", return_value=False),
            patch("src.train._train_catboost_baseline", return_value=dummy),
            patch("src.train._train_xgboost_baseline", return_value=dummy),
            patch("src.train._train_lightgbm_baseline", return_value=dummy),
            patch("src.train._train_gbm_baseline", return_value=dummy),
            patch("src.train._run_optuna_tuning", side_effect=capture_optuna),
            patch("src.train._is_already_registered", return_value=False),
            patch("src.train._register_best_model"),
        ):
            run_training(pipeline_path="/my/pipeline.joblib")

        assert len(captured) == 1
        assert captured[0] == "/my/pipeline.joblib"


# ---------------------------------------------------------------------------
# _log_common_tags
# ---------------------------------------------------------------------------


class TestLogCommonTags:
    def test_sets_model_type_and_python_version(self) -> None:
        from src.train import _log_common_tags

        with patch("src.train.mlflow.set_tag") as mock_tag:
            _log_common_tags("catboost")
            calls = {c.args[0]: c.args[1] for c in mock_tag.call_args_list}
            assert calls["model_type"] == "catboost"
            assert calls["python_version"] == "3.11"

    def test_model_type_passed_through(self) -> None:
        from src.train import _log_common_tags

        with patch("src.train.mlflow.set_tag") as mock_tag:
            _log_common_tags("xgboost")
            calls = {c.args[0]: c.args[1] for c in mock_tag.call_args_list}
            assert calls["model_type"] == "xgboost"


# ---------------------------------------------------------------------------
# _load_data — processed_dir propagation
# ---------------------------------------------------------------------------


class TestLoadData:
    def _make_parquet_df(self, n: int = 20) -> pd.DataFrame:
        from src.constants import LOG_PRICE_COL

        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "Cut": rng.integers(0, 4, n).astype(float),
                "Colour": rng.integers(0, 20, n).astype(float),
                "Clarity": rng.integers(0, 11, n).astype(float),
                "Polish": rng.integers(0, 4, n).astype(float),
                "Symmetry": rng.integers(0, 4, n).astype(float),
                "Fluorescence": rng.integers(0, 7, n).astype(float),
                "Weight": rng.uniform(0.2, 3.0, n),
                "length": rng.uniform(3.0, 8.0, n),
                "width": rng.uniform(3.0, 8.0, n),
                "depth_mm": rng.uniform(2.0, 5.0, n),
                "volume": rng.uniform(10, 200, n),
                "carat_per_volume": rng.uniform(0.001, 0.1, n),
                "log_weight": rng.uniform(0.1, 1.1, n),
                LOG_PRICE_COL: rng.uniform(6.0, 10.0, n),
            }
        )

    def test_returns_correct_shapes(self) -> None:
        from src.train import _load_data

        df = self._make_parquet_df(20)

        mock_pipeline = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.ones((20, 22))
        mock_preprocessor.transformers_ = []
        mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}

        with (
            patch("src.train.pd.read_parquet", return_value=df),
            patch("src.train.load_pipeline", return_value=mock_pipeline),
            patch(
                "src.train._get_feature_names",
                return_value=[f"f{i}" for i in range(22)],
            ),
        ):
            X_train, X_test, y_train, y_test, feature_names = _load_data()

        assert X_train.shape == (20, 22)
        assert len(y_train) == 20
        assert len(feature_names) == 22

    def test_target_not_in_features(self) -> None:
        from src.constants import LOG_PRICE_COL
        from src.train import _load_data

        df = self._make_parquet_df(20)

        mock_pipeline = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.ones((20, 13))
        mock_preprocessor.transformers_ = []
        mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}

        transform_inputs: list[list[str]] = []

        def capture_transform(X: pd.DataFrame) -> np.ndarray:
            transform_inputs.append(list(X.columns))
            return np.ones((len(X), 13))

        mock_preprocessor.transform.side_effect = capture_transform

        with (
            patch("src.train.pd.read_parquet", return_value=df),
            patch("src.train.load_pipeline", return_value=mock_pipeline),
            patch(
                "src.train._get_feature_names",
                return_value=[f"f{i}" for i in range(13)],
            ),
        ):
            _load_data()

        for cols in transform_inputs:
            assert LOG_PRICE_COL not in cols

    def test_custom_processed_dir_used_for_parquet_paths(self) -> None:
        """_load_data(processed_dir='/x') must read from /x/train.parquet,
        not from the constant TRAIN_PARQUET_PATH."""
        from src.constants import TEST_PARQUET_PATH, TRAIN_PARQUET_PATH
        from src.train import _load_data

        df = self._make_parquet_df(10)
        read_paths: list[str] = []

        def capture_read(path: str) -> pd.DataFrame:
            read_paths.append(path)
            return df

        mock_pipeline = MagicMock()
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.ones((10, 5))
        mock_preprocessor.transformers_ = []
        mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}

        with (
            patch("src.train.pd.read_parquet", side_effect=capture_read),
            patch("src.train.load_pipeline", return_value=mock_pipeline),
            patch("src.train._get_feature_names", return_value=["f0"] * 5),
        ):
            _load_data(processed_dir="/custom/processed")

        assert all(p.startswith("/custom/processed") for p in read_paths)
        assert TRAIN_PARQUET_PATH not in read_paths
        assert TEST_PARQUET_PATH not in read_paths


# ---------------------------------------------------------------------------
# Baseline trainer wrappers — each creates the right model type and
# forwards pipeline_path to _train_and_log
# ---------------------------------------------------------------------------


class TestBaselineTrainers:
    """Each _train_*_baseline wrapper should call _train_and_log with
    the correct model class and forward pipeline_path."""

    def _run_trainer(
        self,
        trainer_fn: Callable[..., tuple[str, float]],
        expected_model_cls: type,
    ) -> tuple[list[object], list[object]]:
        """
        Invoke *trainer_fn* with a custom pipeline_path and return
        (captured_models, captured_pipeline_paths).
        """
        dummy: tuple[str, float] = ("run_x", 0.5)
        captured_models: list[object] = []
        captured_paths: list[object] = []

        def capture(*args: object, **kwargs: object) -> tuple[str, float]:
            captured_models.append(args[0])
            captured_paths.append(kwargs.get("pipeline_path", "NOT_PASSED"))
            return dummy

        with patch("src.train._train_and_log", side_effect=capture):
            trainer_fn(
                "exp_1",
                np.ones((10, 5)),
                np.ones((5, 5)),
                np.ones(10),
                np.ones(5),
                ["f1", "f2", "f3", "f4", "f5"],
                pipeline_path="/test/pipeline.joblib",
            )

        assert isinstance(captured_models[0], expected_model_cls)
        return captured_models, captured_paths

    def test_catboost_baseline_model_type(self) -> None:
        from src.train import _train_catboost_baseline

        _, paths = self._run_trainer(_train_catboost_baseline, CatBoostRegressor)
        assert paths[0] == "/test/pipeline.joblib"

    def test_xgboost_baseline_model_type(self) -> None:
        from src.train import _train_xgboost_baseline

        _, paths = self._run_trainer(_train_xgboost_baseline, XGBRegressor)
        assert paths[0] == "/test/pipeline.joblib"

    def test_lightgbm_baseline_model_type(self) -> None:
        from src.train import _train_lightgbm_baseline

        _, paths = self._run_trainer(_train_lightgbm_baseline, LGBMRegressor)
        assert paths[0] == "/test/pipeline.joblib"

    def test_gbm_baseline_model_type(self) -> None:
        from src.train import _train_gbm_baseline

        _, paths = self._run_trainer(_train_gbm_baseline, GradientBoostingRegressor)
        assert paths[0] == "/test/pipeline.joblib"


# ---------------------------------------------------------------------------
# _train_and_log — extra_params branch + pipeline exists branch
# ---------------------------------------------------------------------------


class TestTrainAndLogExtended:
    def _make_mock_model(self) -> MagicMock:
        model = MagicMock()
        model.get_params.return_value = {"n_estimators": 100}
        model.predict.return_value = np.array([1.0, 2.0, 3.0])
        return model

    def test_extra_params_logged(self) -> None:
        """extra_params dict should be merged into logged params."""
        from src.train import _train_and_log

        mock_model = self._make_mock_model()
        y = np.array([1.0, 2.0, 3.0])
        mock_run = MagicMock()
        mock_run.info.run_id = "r1"

        logged: dict[str, object] = {}

        def capture_params(p: dict[str, object]) -> None:
            logged.update(p)

        with (
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params", side_effect=capture_params),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch("src.train.Path.exists", return_value=False),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)
            _train_and_log(
                mock_model,
                "catboost",
                "run",
                "exp",
                np.ones((3, 2)),
                np.ones((3, 2)),
                y,
                y,
                ["a", "b"],
                extra_params={"best_depth": 6},
            )

        assert "best_depth" in logged

    def test_pipeline_artifact_logged_when_exists(self) -> None:
        """Pipeline should be logged as artifact when file exists."""
        from src.train import _train_and_log

        mock_model = self._make_mock_model()
        y = np.array([1.0, 2.0, 3.0])
        mock_run = MagicMock()
        mock_run.info.run_id = "r2"

        artifact_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with (
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch(
                "src.train.mlflow.log_artifact",
                side_effect=lambda *a, **kw: artifact_calls.append((a, kw)),
            ),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch("src.train.Path.exists", return_value=True),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)
            _train_and_log(
                mock_model,
                "catboost",
                "run",
                "exp",
                np.ones((3, 2)),
                np.ones((3, 2)),
                y,
                y,
                ["a", "b"],
            )

        # At least two artifact calls: pipeline + selected_features.json
        assert len(artifact_calls) >= 2


# ---------------------------------------------------------------------------
# _run_optuna_tuning — mocked to avoid real training
# ---------------------------------------------------------------------------


class TestRunOptunaTuning:
    def _make_study(self, best_rmse: float = 0.35) -> MagicMock:
        trial = MagicMock()
        trial.number = 3
        study = MagicMock()
        study.best_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
        }
        study.best_value = best_rmse
        study.best_trial = trial
        return study

    def test_returns_run_id_and_rmse(self) -> None:
        from src.train import _run_optuna_tuning

        mock_run = MagicMock()
        mock_run.info.run_id = "tuned_run"
        study = self._make_study(0.35)

        mock_final_model = MagicMock()
        mock_final_model.predict.return_value = np.ones(5)

        with (
            patch("src.train.optuna.logging.set_verbosity"),
            patch("src.train.MLflowCallback"),
            patch("src.train.optuna.create_study", return_value=study),
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch(
                "src.train.mlflow.get_tracking_uri",
                return_value="http://localhost:5000",
            ),
            patch("src.train.CatBoostRegressor", return_value=mock_final_model),
            patch("src.train.Path.exists", return_value=False),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            run_id, rmse = _run_optuna_tuning(
                "exp_1",
                np.ones((10, 4)),
                np.ones((5, 4)),
                np.ones(10),
                np.ones(5),
                ["f1", "f2", "f3", "f4"],
                n_trials=2,
            )

        assert run_id == "tuned_run"
        assert isinstance(rmse, float)

    def test_custom_pipeline_path_used(self) -> None:
        """pipeline_path kwarg must reach the Path.exists / log_artifact call,
        not fall back to PIPELINE_PATH constant."""
        from src.train import _run_optuna_tuning

        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        study = self._make_study()
        mock_final_model = MagicMock()
        mock_final_model.predict.return_value = np.ones(5)

        artifact_calls: list[str] = []

        with (
            patch("src.train.optuna.logging.set_verbosity"),
            patch("src.train.MLflowCallback"),
            patch("src.train.optuna.create_study", return_value=study),
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch(
                "src.train.mlflow.log_artifact",
                side_effect=lambda *a, **kw: artifact_calls.append(a[0]),
            ),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch(
                "src.train.mlflow.get_tracking_uri",
                return_value="http://localhost:5000",
            ),
            patch("src.train.CatBoostRegressor", return_value=mock_final_model),
            patch("src.train.Path.exists", return_value=True),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            _run_optuna_tuning(
                "exp_1",
                np.ones((10, 4)),
                np.ones((5, 4)),
                np.ones(10),
                np.ones(5),
                ["f1", "f2", "f3", "f4"],
                n_trials=2,
                pipeline_path="/custom/pipeline.joblib",
            )

        assert "/custom/pipeline.joblib" in artifact_calls

    def test_study_optimized_with_n_trials(self) -> None:
        from src.train import _run_optuna_tuning

        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        study = self._make_study()
        mock_final_model = MagicMock()
        mock_final_model.predict.return_value = np.ones(5)

        with (
            patch("src.train.optuna.logging.set_verbosity"),
            patch("src.train.MLflowCallback"),
            patch("src.train.optuna.create_study", return_value=study),
            patch("src.train.mlflow.start_run") as mock_start,
            patch("src.train.mlflow.log_params"),
            patch("src.train.mlflow.log_metrics"),
            patch("src.train.mlflow.sklearn.log_model"),
            patch("src.train.mlflow.log_artifact"),
            patch("src.train.mlflow.set_tag"),
            patch("src.train.infer_signature"),
            patch(
                "src.train.mlflow.get_tracking_uri",
                return_value="http://localhost:5000",
            ),
            patch("src.train.CatBoostRegressor", return_value=mock_final_model),
            patch("src.train.Path.exists", return_value=False),
        ):
            mock_start.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            _run_optuna_tuning(
                "exp_1",
                np.ones((10, 4)),
                np.ones((5, 4)),
                np.ones(10),
                np.ones(5),
                ["f1", "f2", "f3", "f4"],
                n_trials=7,
            )

        all_args, all_kwargs = study.optimize.call_args
        n_trials_used = all_kwargs.get("n_trials") or (
            all_args[1] if len(all_args) > 1 else None
        )
        assert n_trials_used == 7


# ---------------------------------------------------------------------------
# _verify_model_loads
# ---------------------------------------------------------------------------


class TestVerifyModelLoads:
    def test_calls_load_model_with_correct_uri(self) -> None:
        from src.train import _verify_model_loads

        with patch("mlflow.pyfunc.load_model") as mock_load:
            _verify_model_loads("Diamond", "Production")
            mock_load.assert_called_once_with("models:/Diamond/Production")

    def test_different_stage(self) -> None:
        from src.train import _verify_model_loads

        with patch("mlflow.pyfunc.load_model") as mock_load:
            _verify_model_loads("Diamond", "Staging")
            mock_load.assert_called_once_with("models:/Diamond/Staging")
