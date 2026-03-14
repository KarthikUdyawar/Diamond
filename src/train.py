"""
src/train.py
Model training pipeline for the Diamond price prediction project.

Public API:
    run_training()   — orchestrator: load data → feature select → train 4
                       baselines → Optuna tuning → register best model
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from optuna_integration.mlflow import MLflowCallback
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.constants import (
    MLFLOW_EXPERIMENT_NAME,
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
from src.features import load_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

_PYTHON_VERSION = "3.11"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, R², MAPE on log-space predictions vs actuals."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # MAPE — guard against near-zero actuals
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# MLflow experiment helpers
# ---------------------------------------------------------------------------


def _get_or_create_experiment(name: str) -> str:
    """
    Return experiment ID, creating the experiment if it doesn't exist.

    Artifacts are proxied via the MLflow server (--serve-artifacts), so no
    local artifact_root is needed. If a soft-deleted experiment with the same
    name exists it is restored (Postgres unique-name constraint prevents
    re-creation). Run `make reset-mlflow` to fully wipe the DB and start clean.
    """
    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is not None:
        exp_id = experiment.experiment_id
        if experiment.lifecycle_stage == "deleted":
            logger.warning("Experiment '%s' was deleted — restoring it.", name)
            client.restore_experiment(exp_id)
        logger.info("Using existing MLflow experiment '%s' (id=%s).", name, exp_id)
        return cast(str, exp_id)

    experiment_id = mlflow.create_experiment(name)
    logger.info("Created MLflow experiment '%s' (id=%s).", name, experiment_id)
    return cast(str, experiment_id)


def _run_exists(experiment_id: str, run_name: str) -> bool:
    """Return True if a run with *run_name* already exists in *experiment_id*."""
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    return len(runs) > 0


def _log_common_tags(model_type: str) -> None:
    """Log shared tags on the active MLflow run."""
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("python_version", _PYTHON_VERSION)


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

    Parquet paths are derived from *processed_dir* so that callers who pass a
    custom directory actually get data from that directory (previously the paths
    were hardcoded to the constants, making the *processed_dir* argument a
    no-op).

    Returns
    -------
    X_train_t, X_test_t : np.ndarray
        Transformed feature matrices.
    y_train, y_test : np.ndarray
        Log-price target vectors.
    feature_names : list[str]
        Output feature names from the ColumnTransformer.
    """
    from src.constants import LOG_PRICE_COL

    # Derive parquet paths from processed_dir so the argument is honoured.
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

    # Derive output feature names
    feature_names = _get_feature_names(preprocessor)
    logger.info(
        "Data loaded — train: %s, test: %s, features: %d.",
        X_train_t.shape,
        X_test_t.shape,
        len(feature_names),
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
    logger.info("Selected %d features: %s", n_features, selected_names)
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
) -> tuple[str, float]:
    """
    Train *model*, log everything to MLflow, return (run_id, rmse).

    Parameters
    ----------
    extra_params:
        Additional params to log beyond what the model exposes (e.g. Optuna
        best params that aren't in model.get_params()).
    pipeline_path:
        Path to the fitted pipeline joblib to attach as an MLflow artifact.
        Callers should always pass this explicitly so the logged artifact
        matches the pipeline that actually produced the features for this run.
    """
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        _log_common_tags(model_type)

        # Log hyperparameters
        params = model.get_params() if hasattr(model, "get_params") else {}
        if extra_params:
            params.update(extra_params)
        params["n_features_selected"] = len(selected_names)
        params["train_size"] = len(X_train)
        params["test_size"] = len(X_test)
        mlflow.log_params({str(k): str(v) for k, v in params.items()})

        # Train
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)
        metrics["training_time_sec"] = round(training_time, 3)
        mlflow.log_metrics(metrics)

        # Log model artifact with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Log pipeline artifact — uses the caller-supplied path, never the global
        if Path(pipeline_path).exists():
            mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        # Log selected feature names
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(selected_names, tmp, indent=2)
            features_path = tmp.name
        mlflow.log_artifact(features_path)

        run_id = run.info.run_id
        logger.info(
            "Run '%s' complete — RMSE=%.4f  R²=%.4f  time=%.1fs",
            run_name,
            metrics["rmse"],
            metrics["r2"],
            training_time,
        )
        return run_id, metrics["rmse"]


def _train_catboost_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
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
    )


def _train_xgboost_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
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
    )


def _train_lightgbm_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
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
    )


def _train_gbm_baseline(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    pipeline_path: str = PIPELINE_PATH,
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
    )


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def _run_optuna_tuning(
    experiment_id: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_names: list[str],
    n_trials: int = OPTUNA_N_TRIALS,
    pipeline_path: str = PIPELINE_PATH,
) -> tuple[str, float]:
    """
    Run Optuna HPO on CatBoost, each trial logged as a nested MLflow run.
    Re-trains the final model with best params and logs as *catboost-tuned*.

    Parameters
    ----------
    pipeline_path:
        Path to the fitted pipeline joblib to attach as an MLflow artifact.
        Forwarded from run_training() so the logged artifact always matches
        the pipeline that produced the features for this run.

    Returns
    -------
    run_id : str
    rmse : float
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow_cb = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="rmse",
        mlflow_kwargs={"experiment_id": experiment_id, "nested": True},
    )

    # Carve a fixed validation split out of the training data.
    # X_test / y_test remain completely untouched until the final one-time
    # evaluation of the best model after HPO is complete.
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
            "depth": trial.suggest_int(
                "depth", OPTUNA_CB_DEPTH_LOW, OPTUNA_CB_DEPTH_HIGH
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", OPTUNA_CB_L2_LOW, OPTUNA_CB_L2_HIGH
            ),
            "subsample": trial.suggest_float(
                "subsample", OPTUNA_CB_SUBSAMPLE_LOW, OPTUNA_CB_SUBSAMPLE_HIGH
            ),
        }
        model = CatBoostRegressor(
            **params,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        return rmse

    # Run study — wrap in a parent run so trials nest under catboost-tuned
    with mlflow.start_run(
        run_name=RUN_NAME_CATBOOST_TUNED, experiment_id=experiment_id
    ) as parent_run:
        _log_common_tags("catboost")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

        best_params = study.best_params
        logger.info("Optuna best params: %s", best_params)
        logger.info("Optuna best RMSE: %.4f", study.best_value)

        # Re-train final model with best params
        final_model = CatBoostRegressor(
            **best_params,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
        t0 = time.perf_counter()
        final_model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

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

        # Log model artifact with signature
        signature = infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, "model", signature=signature)

        # Log pipeline artifact — uses the caller-supplied path, never the global
        if Path(pipeline_path).exists():
            mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(selected_names, tmp, indent=2)
            features_path = tmp.name
        mlflow.log_artifact(features_path)

        run_id = parent_run.info.run_id
        logger.info(
            "Tuned run '%s' complete — RMSE=%.4f  R²=%.4f  time=%.1fs",
            RUN_NAME_CATBOOST_TUNED,
            metrics["rmse"],
            metrics["r2"],
            training_time,
        )
        return run_id, metrics["rmse"]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def _is_already_registered(
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    stage: str = MLFLOW_MODEL_STAGE,
) -> bool:
    """
    Return True if a model version in *stage* already points to *run_id*.

    Prevents mlflow.register_model() from creating a duplicate version on
    every pipeline re-run when all training runs were skipped (idempotency
    guard for run_training()).
    """
    client = mlflow.MlflowClient()
    try:
        staged = client.get_latest_versions(model_name, stages=[stage])
        return any(mv.run_id == run_id for mv in staged)
    except Exception:
        # Model doesn't exist in registry yet — safe to register.
        return False


def _verify_model_loads(model_name: str, stage: str) -> None:
    """Load the registered model from the registry as a smoke test."""
    production_uri = f"models:/{model_name}/{stage}"
    loaded = mlflow.pyfunc.load_model(production_uri)
    logger.info(
        "Verified: loaded model from '%s' — type: %s.",
        production_uri,
        type(loaded),
    )


def _register_best_model(
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    stage: str = MLFLOW_MODEL_STAGE,
) -> None:
    """
    Register the model from *run_id* in the MLflow Model Registry and
    promote it to *stage*.
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    # Register — creates version 1 if model doesn't exist yet
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = model_version.version
    logger.info("Registered model '%s' version %s.", model_name, version)

    # Promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("Model '%s' v%s promoted to '%s'.", model_name, version, stage)

    # Verify the model loads
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

    1. Load transformed train/test data from *processed_dir* using the
       pipeline at *pipeline_path*.
    2. CatBoost-based feature selection → top 14 features.
    3. Train 4 baselines (skip each if a run with that name already exists).
    4. Run Optuna trials on CatBoost (skip if catboost-tuned already exists).
    5. Register the best run (lowest RMSE) as Diamond/Production.

    Both *processed_dir* and *pipeline_path* are forwarded all the way through
    to _load_data, every baseline wrapper, and _run_optuna_tuning so that the
    MLflow artifact logged for each run is always the same pipeline that
    produced the features for that run.
    """
    load_dotenv(override=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    experiment_id = _get_or_create_experiment(experiment_name)

    # Load data — both processed_dir and pipeline_path are honoured here
    X_train, X_test, y_train, y_test, feature_names = _load_data(
        processed_dir=processed_dir,
        pipeline_path=pipeline_path,
    )

    # Feature selection
    selected_indices, selected_names = _select_features(X_train, y_train, feature_names)
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    # Baseline runs — skip if already exist
    results: dict[str, tuple[str, float]] = {}

    baseline_trainers = [
        (RUN_NAME_CATBOOST, _train_catboost_baseline),
        (RUN_NAME_XGBOOST, _train_xgboost_baseline),
        (RUN_NAME_LIGHTGBM, _train_lightgbm_baseline),
        (RUN_NAME_GBM, _train_gbm_baseline),
    ]

    for run_name, trainer_fn in baseline_trainers:
        if _run_exists(experiment_id, run_name):
            logger.info("Skipping '%s' — run already exists.", run_name)
            client = mlflow.MlflowClient()
            existing = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1,
            )
            if existing:
                run_id = existing[0].info.run_id
                rmse = existing[0].data.metrics.get("rmse", float("inf"))
                results[run_name] = (run_id, rmse)
        else:
            run_id, rmse = trainer_fn(
                experiment_id,
                X_train_sel,
                X_test_sel,
                y_train,
                y_test,
                selected_names,
                pipeline_path=pipeline_path,  # forwarded — never falls back to constant
            )
            results[run_name] = (run_id, rmse)

    # Optuna tuning — skip if already exists
    if _run_exists(experiment_id, RUN_NAME_CATBOOST_TUNED):
        logger.info("Skipping '%s' — run already exists.", RUN_NAME_CATBOOST_TUNED)
        client = mlflow.MlflowClient()
        existing = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = '{RUN_NAME_CATBOOST_TUNED}'",
            max_results=1,
        )
        if existing:
            run_id = existing[0].info.run_id
            rmse = existing[0].data.metrics.get("rmse", float("inf"))
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
            pipeline_path=pipeline_path,  # forwarded — never falls back to constant
        )
        results[RUN_NAME_CATBOOST_TUNED] = (run_id, rmse)

    # Find best run
    best_run_name = min(results, key=lambda k: results[k][1])
    best_run_id, best_rmse = results[best_run_name]
    logger.info(
        "Best run: '%s' (run_id=%s, RMSE=%.4f).",
        best_run_name,
        best_run_id,
        best_rmse,
    )

    # Register — skip if the staged version already points to this run
    if _is_already_registered(best_run_id):
        logger.info(
            "Skipping registration — '%s/%s' already points to run_id=%s.",
            MLFLOW_MODEL_NAME,
            MLFLOW_MODEL_STAGE,
            best_run_id,
        )
        _verify_model_loads(MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE)
    else:
        _register_best_model(best_run_id)
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run_training()
