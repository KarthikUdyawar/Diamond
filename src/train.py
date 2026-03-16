"""
src/train.py
Model training pipeline for the Diamond price prediction project.

Public API:
    run_training()   — orchestrator: load data → feature select → train 4
                       baselines → Optuna tuning → register best model
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from optuna_integration.mlflow import MLflowCallback
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import get_settings
from src.constants import (
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_STAGE,
    N_FEATURES_TO_SELECT,
    OPTUNA_CB_DEPTH_HIGH,
    OPTUNA_CB_DEPTH_LOW,
    OPTUNA_CB_ITERATIONS_HIGH,
    OPTUNA_CB_ITERATIONS_LOW,
    OPTUNA_CB_L2_HIGH,
    OPTUNA_CB_L2_LOW,
    OPTUNA_CB_LR_HIGH,
    OPTUNA_CB_LR_LOW,
    OPTUNA_CB_SUBSAMPLE_HIGH,
    OPTUNA_CB_SUBSAMPLE_LOW,
    OPTUNA_N_TRIALS,
    PIPELINE_PATH,
    PROCESSED_DIR,
    RANDOM_STATE,
    RUN_NAME_CATBOOST,
    RUN_NAME_CATBOOST_TUNED,
    RUN_NAME_GBM,
    RUN_NAME_LIGHTGBM,
    RUN_NAME_XGBOOST,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
)
from src.explain import ExplainerService, init_explainer
from src.features import load_pipeline
from src.logger import bind_context, clear_context, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

_PYTHON_VERSION = "3.11"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, R², MAPE on log-space predictions vs actuals."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mask = np.abs(y_true) > 1e-6
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0
    logger.debug("metrics_computed", rmse=rmse, mae=mae, r2=r2, mape=mape)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# MLflow experiment helpers
# ---------------------------------------------------------------------------


def _get_or_create_experiment(name: str) -> str:
    """
    Return experiment ID, creating the experiment if it doesn't exist.

    Restores soft-deleted experiments transparently.
    """
    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        exp_id = experiment.experiment_id
        if experiment.lifecycle_stage == "deleted":
            logger.warning("experiment_was_deleted_restoring", name=name)
            client.restore_experiment(exp_id)
        logger.info("experiment_found", name=name, experiment_id=exp_id)
        return str(exp_id)

    experiment_id = mlflow.create_experiment(name)
    logger.info("experiment_created", name=name, experiment_id=experiment_id)
    return str(experiment_id)


def _run_exists(
    experiment_id: str,
    run_name: str,
    run_scope: dict[str, str] | None = None,
) -> bool:
    """
    Return True if a matching run already exists in *experiment_id*.

    Matching is keyed by *run_name* plus optional *run_scope* tags.
    Tag keys containing dots are backtick-quoted for MLflow search grammar.
    """
    client = mlflow.MlflowClient()
    filters = [f"tags.mlflow.runName = '{run_name}'"]
    if run_scope:
        for k, v in run_scope.items():
            safe_key = f"`{k}`" if "." in k else k
            filters.append(f"tags.{safe_key} = '{v}'")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=" and ".join(filters),
        max_results=1,
    )
    exists = len(runs) > 0
    if exists:
        logger.info("run_skipped_already_exists", run_name=run_name)
    return exists


def _log_common_tags(
    model_type: str,
    run_scope: dict[str, str] | None = None,
) -> None:
    """Log shared tags on the active MLflow run, including provenance scope."""
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("python_version", _PYTHON_VERSION)
    if run_scope:
        for k, v in run_scope.items():
            mlflow.set_tag(k, v)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(
    processed_dir: str = PROCESSED_DIR,
    pipeline_path: str = PIPELINE_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load processed parquet files from *processed_dir* and apply the fitted
    pipeline at *pipeline_path*.

    Returns
    -------
    X_train_t, X_test_t : np.ndarray
    y_train, y_test : np.ndarray
    feature_names : list[str]
    """
    from src.constants import LOG_PRICE_COL

    processed = Path(processed_dir)
    train_path = str(processed / Path(TRAIN_PARQUET_PATH).name)
    test_path = str(processed / Path(TEST_PARQUET_PATH).name)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    y_train = train_df[LOG_PRICE_COL].to_numpy()
    y_test = test_df[LOG_PRICE_COL].to_numpy()

    X_train = train_df.drop(columns=[LOG_PRICE_COL])
    X_test = test_df.drop(columns=[LOG_PRICE_COL])

    pipeline = load_pipeline(pipeline_path)
    preprocessor = pipeline.named_steps["preprocessor"]

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = _get_feature_names(preprocessor)
    logger.info(
        "data_loaded",
        train_shape=X_train_t.shape,
        test_shape=X_test_t.shape,
        n_features=len(feature_names),
    )
    return X_train_t, X_test_t, y_train, y_test, feature_names


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract output feature names from the fitted ColumnTransformer."""
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "ordinal":
            names.extend(cols)
        elif name == "nominal":
            ohe = transformer.named_steps["encoder"]
            names.extend([f"Shape_{c}" for c in ohe.categories_[0]])
        elif name == "numeric":
            names.extend(cols)
    return names


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_features: int = N_FEATURES_TO_SELECT,
) -> tuple[list[int], list[str]]:
    """
    Use a CatBoostRegressor to rank features and return the top *n_features*.

    Returns
    -------
    selected_indices : list[int]
    selected_names : list[str]
    """
    selector = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    selector.fit(X_train, y_train)
    importances: np.ndarray = np.asarray(selector.get_feature_importance())
    ranked = np.argsort(importances)[::-1]
    selected_indices = sorted(ranked[:n_features].tolist())
    selected_names = [feature_names[i] for i in selected_indices]
    logger.info(
        "feature_selection_done",
        n_selected=n_features,
        selected=selected_names,
    )
    return selected_indices, selected_names


# ---------------------------------------------------------------------------
# Baseline trainers
# ---------------------------------------------------------------------------


def _train_and_log(
    model: Any,  # noqa: ANN401
    model_type: str,
    run_name: str,
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    extra_params: dict[str, Any] | None = None,
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    """
    Train *model*, log everything to MLflow, return (run_id, rmse).
    """
    bind_context(run_name=run_name, model_type=model_type)
    logger.info("training_started", run_name=run_name)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        _log_common_tags(model_type, run_scope=run_scope)

        params = model.get_params() if hasattr(model, "get_params") else {}
        if extra_params:
            params.update(extra_params)
        params["n_features_selected"] = len(selected_names)
        params["train_size"] = len(X_train)
        params["test_size"] = len(X_test)
        mlflow.log_params({str(k): str(v) for k, v in params.items()})

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        metrics["training_time_sec"] = round(training_time, 3)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        if Path(pipeline_path).exists():
            mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = Path(tmpdir) / "selected_features.json"
            features_path.write_text(json.dumps(selected_names, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(features_path))

        run_id = run.info.run_id

    logger.info(
        "training_done",
        run_name=run_name,
        run_id=run_id,
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        elapsed_sec=round(training_time, 2),
    )
    clear_context()
    return run_id, metrics["rmse"]


def _train_catboost_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    return _train_and_log(
        model,
        "catboost",
        RUN_NAME_CATBOOST,
        experiment_id,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_names,
        pipeline_path=pipeline_path,
        run_scope=run_scope,
    )


def _train_xgboost_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return _train_and_log(
        model,
        "xgboost",
        RUN_NAME_XGBOOST,
        experiment_id,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_names,
        pipeline_path=pipeline_path,
        run_scope=run_scope,
    )


def _train_lightgbm_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    return _train_and_log(
        model,
        "lightgbm",
        RUN_NAME_LIGHTGBM,
        experiment_id,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_names,
        pipeline_path=pipeline_path,
        run_scope=run_scope,
    )


def _train_gbm_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    return _train_and_log(
        model,
        "gbm",
        RUN_NAME_GBM,
        experiment_id,
        X_train,
        X_test,
        y_train,
        y_test,
        selected_names,
        pipeline_path=pipeline_path,
        run_scope=run_scope,
    )


# ---------------------------------------------------------------------------
# Optuna tuning + SHAP
# ---------------------------------------------------------------------------


def _log_shap_artifacts(
    final_model: CatBoostRegressor,
    X_train: np.ndarray,
    selected_names: list[str],
) -> ExplainerService:
    """
    Build ExplainerService and log global SHAP plots as MLflow artifacts.

    Must be called inside an active mlflow.start_run() context.
    Only called for the tuned model (best model only — per plan decision Q4).

    Returns the initialised ExplainerService so it is available for the API.
    """
    X_train_df = pd.DataFrame(X_train, columns=selected_names)

    explainer = init_explainer(final_model, X_train_df, selected_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger.info("shap_artifacts_generating", output_dir=tmpdir)

        summary_path = explainer.plot_summary(X_train_df, output_dir=tmp_path)
        dependence_path = explainer.plot_dependence(
            X_train_df, feature="Weight", output_dir=tmp_path
        )
        importance_path = explainer.plot_importance(X_train_df, output_dir=tmp_path)

        mlflow.log_artifact(str(summary_path), artifact_path="shap")
        mlflow.log_artifact(str(dependence_path), artifact_path="shap")
        mlflow.log_artifact(str(importance_path), artifact_path="shap")

    logger.info("shap_artifacts_logged")
    return explainer


def _run_optuna_tuning(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    n_trials: int = OPTUNA_N_TRIALS,
    pipeline_path: str = PIPELINE_PATH,
    run_scope: dict[str, str] | None = None,
) -> tuple[str, float]:
    """
    Run Optuna HPO on CatBoost, each trial logged as a nested MLflow run.
    Re-trains the final model with best params, logs SHAP artifacts, and
    returns (run_id, rmse).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="rmse",
        mlflow_kwargs={"experiment_id": experiment_id, "nested": True},
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int(
                "iterations", OPTUNA_CB_ITERATIONS_LOW, OPTUNA_CB_ITERATIONS_HIGH
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", OPTUNA_CB_LR_LOW, OPTUNA_CB_LR_HIGH, log=True
            ),
            "depth": trial.suggest_int("depth", OPTUNA_CB_DEPTH_LOW, OPTUNA_CB_DEPTH_HIGH),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", OPTUNA_CB_L2_LOW, OPTUNA_CB_L2_HIGH),
            "subsample": trial.suggest_float(
                "subsample", OPTUNA_CB_SUBSAMPLE_LOW, OPTUNA_CB_SUBSAMPLE_HIGH
            ),
        }
        model = CatBoostRegressor(**params, random_seed=RANDOM_STATE, verbose=0)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, y_pred)))

    bind_context(run_name=RUN_NAME_CATBOOST_TUNED)
    logger.info("optuna_tuning_started", n_trials=n_trials)

    with mlflow.start_run(
        run_name=RUN_NAME_CATBOOST_TUNED, experiment_id=experiment_id
    ) as parent_run:
        _log_common_tags("catboost", run_scope=run_scope)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

        best_params = study.best_params
        logger.info(
            "optuna_tuning_done",
            best_rmse=study.best_value,
            best_trial=study.best_trial.number,
            best_params=best_params,
        )

        # Re-train final model on full training data with best params
        final_model = CatBoostRegressor(**best_params, random_seed=RANDOM_STATE, verbose=0)
        t0 = time.perf_counter()
        final_model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        # One-time evaluation on the held-out test set
        y_pred = final_model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        metrics["training_time_sec"] = round(training_time, 3)
        metrics["optuna_best_trial"] = study.best_trial.number

        all_params: dict[str, Any] = {**best_params}
        all_params["n_features_selected"] = len(selected_names)
        all_params["train_size"] = len(X_train)
        all_params["optuna_val_size"] = len(X_val)
        all_params["test_size"] = len(X_test)
        all_params["optuna_n_trials"] = n_trials

        mlflow.log_params({str(k): str(v) for k, v in all_params.items()})
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, "model", signature=signature)

        if Path(pipeline_path).exists():
            mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = Path(tmpdir) / "selected_features.json"
            features_path.write_text(json.dumps(selected_names, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(features_path))

        # Log SHAP artifacts inside the parent run context (best model only)
        _log_shap_artifacts(final_model, X_train, selected_names)

        run_id = parent_run.info.run_id

    logger.info(
        "tuned_run_complete",
        run_id=run_id,
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        elapsed_sec=round(training_time, 2),
    )
    clear_context()
    return run_id, metrics["rmse"]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _is_already_registered(
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    stage: str = MLFLOW_MODEL_STAGE,
) -> bool:
    """Return True if a model version in *stage* already points to *run_id*."""
    client = mlflow.MlflowClient()
    try:
        staged = client.get_latest_versions(model_name, stages=[stage])
        return any(mv.run_id == run_id for mv in staged)
    except mlflow.exceptions.MlflowException as exc:
        error_code = getattr(exc, "error_code", None)
        if error_code == "RESOURCE_DOES_NOT_EXIST" or "RESOURCE_DOES_NOT_EXIST" in str(exc):
            logger.debug("model_not_yet_in_registry", name=model_name)
            return False
        raise


def _verify_model_loads(model_name: str, stage: str) -> None:
    """Load the registered model from the registry as a smoke test."""
    production_uri = f"models:/{model_name}/{stage}"
    loaded = mlflow.pyfunc.load_model(production_uri)
    logger.info("model_load_verified", uri=production_uri, type=str(type(loaded)))


def _register_best_model(
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    stage: str = MLFLOW_MODEL_STAGE,
) -> None:
    """Register the model from *run_id* and promote it to *stage*."""
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = model_version.version
    logger.info("model_registered", name=model_name, version=version)

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("model_promoted", name=model_name, version=version, stage=stage)

    _verify_model_loads(model_name, stage)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_training(
    processed_dir: str = PROCESSED_DIR,
    pipeline_path: str = PIPELINE_PATH,
    n_trials: int = OPTUNA_N_TRIALS,
) -> None:
    """
    Full training orchestrator.

    1. Load transformed train/test data from *processed_dir*.
    2. CatBoost-based feature selection → top 14 features.
    3. Train 4 baselines (skip if run already exists with same provenance).
    4. Run Optuna tuning on CatBoost + log SHAP artifacts.
    5. Register the best run (lowest RMSE) as Diamond/Production.
    """
    cfg = get_settings()
    bind_context(app_env=cfg.app_env, experiment=cfg.mlflow_experiment_name)
    logger.info("training_pipeline_started", env=cfg.app_env)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    logger.info("mlflow_tracking_uri_set", uri=cfg.mlflow_tracking_uri)

    experiment_id = _get_or_create_experiment(cfg.mlflow_experiment_name)

    baseline_scope: dict[str, str] = {
        "data.processed_dir": str(Path(processed_dir).resolve()),
        "data.pipeline_path": str(Path(pipeline_path).resolve()),
    }
    tuning_scope: dict[str, str] = {
        **baseline_scope,
        "train.n_trials": str(n_trials),
    }

    X_train, X_test, y_train, y_test, feature_names = _load_data(
        processed_dir=processed_dir,
        pipeline_path=pipeline_path,
    )

    selected_indices, selected_names = _select_features(X_train, y_train, feature_names)
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    results: dict[str, tuple[str, float]] = {}

    baseline_trainers = [
        (RUN_NAME_CATBOOST, _train_catboost_baseline),
        (RUN_NAME_XGBOOST, _train_xgboost_baseline),
        (RUN_NAME_LIGHTGBM, _train_lightgbm_baseline),
        (RUN_NAME_GBM, _train_gbm_baseline),
    ]

    for run_name, trainer_fn in baseline_trainers:
        if _run_exists(experiment_id, run_name, run_scope=baseline_scope):
            client = mlflow.MlflowClient()
            existing = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=(
                    f"tags.mlflow.runName = '{run_name}' and "
                    f"tags.`data.processed_dir` = '{baseline_scope['data.processed_dir']}' and "
                    f"tags.`data.pipeline_path` = '{baseline_scope['data.pipeline_path']}'"
                ),
                max_results=1,
            )
            if existing:
                run_id = existing[0].info.run_id
                rmse = existing[0].data.metrics.get("rmse", float("inf"))
                results[run_name] = (run_id, rmse)
            else:
                logger.warning(
                    "run_reported_existing_but_not_found_retraining",
                    run_name=run_name,
                )
                run_id, rmse = trainer_fn(
                    experiment_id,
                    X_train_sel,
                    X_test_sel,
                    y_train,
                    y_test,
                    selected_names,
                    pipeline_path=pipeline_path,
                    run_scope=baseline_scope,
                )
                results[run_name] = (run_id, rmse)
        else:
            run_id, rmse = trainer_fn(
                experiment_id,
                X_train_sel,
                X_test_sel,
                y_train,
                y_test,
                selected_names,
                pipeline_path=pipeline_path,
                run_scope=baseline_scope,
            )
            results[run_name] = (run_id, rmse)

    if _run_exists(experiment_id, RUN_NAME_CATBOOST_TUNED, run_scope=tuning_scope):
        client = mlflow.MlflowClient()
        existing = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.mlflow.runName = '{RUN_NAME_CATBOOST_TUNED}' and "
                f"tags.`data.processed_dir` = '{tuning_scope['data.processed_dir']}' and "
                f"tags.`data.pipeline_path` = '{tuning_scope['data.pipeline_path']}' and "
                f"tags.`train.n_trials` = '{tuning_scope['train.n_trials']}'"
            ),
            max_results=1,
        )
        if existing:
            run_id = existing[0].info.run_id
            rmse = existing[0].data.metrics.get("rmse", float("inf"))
            results[RUN_NAME_CATBOOST_TUNED] = (run_id, rmse)
        else:
            logger.warning(
                "tuning_run_reported_existing_but_not_found_retuning",
                run_name=RUN_NAME_CATBOOST_TUNED,
            )
            run_id, rmse = _run_optuna_tuning(
                experiment_id,
                X_train_sel,
                X_test_sel,
                y_train,
                y_test,
                selected_names,
                n_trials=n_trials,
                pipeline_path=pipeline_path,
                run_scope=tuning_scope,
            )
            results[RUN_NAME_CATBOOST_TUNED] = (run_id, rmse)
    else:
        run_id, rmse = _run_optuna_tuning(
            experiment_id,
            X_train_sel,
            X_test_sel,
            y_train,
            y_test,
            selected_names,
            n_trials=n_trials,
            pipeline_path=pipeline_path,
            run_scope=tuning_scope,
        )
        results[RUN_NAME_CATBOOST_TUNED] = (run_id, rmse)

    if not results:
        raise RuntimeError(
            "No runs were collected. Check MLflow server state and experiment integrity."
        )

    best_run_name = min(results, key=lambda k: results[k][1])
    best_run_id, best_rmse = results[best_run_name]
    logger.info(
        "best_run_identified",
        run_name=best_run_name,
        run_id=best_run_id,
        rmse=best_rmse,
    )

    if _is_already_registered(best_run_id):
        logger.info(
            "registration_skipped_already_registered",
            model_name=MLFLOW_MODEL_NAME,
            stage=MLFLOW_MODEL_STAGE,
            run_id=best_run_id,
        )
        _verify_model_loads(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
    else:
        _register_best_model(best_run_id)

    logger.info("training_pipeline_complete")
    clear_context()


if __name__ == "__main__":
    run_training()
