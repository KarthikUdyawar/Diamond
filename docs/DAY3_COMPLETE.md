# 💎 Diamond — Day 3 Complete

> **Model Training + MLflow + Optuna**
> Completed: March 13, 2026

---

## What Was Built

### `src/train.py` — Full training orchestrator

Public API: `run_training()` — loads data → feature selection → 4 baselines → Optuna tuning → model registry.

**Modules implemented:**

- `_compute_metrics()` — RMSE, MAE, R², MAPE on log-space predictions
- `_get_or_create_experiment()` — idempotent experiment setup; restores soft-deleted experiments
- `_run_exists()` — skip-if-exists guard on all runs (safe to re-run `make train`)
- `_load_data()` — loads `train.parquet` + `test.parquet`, applies fitted pipeline, derives feature names
- `_get_feature_names()` — extracts output names from the fitted `ColumnTransformer`
- `_select_features()` — CatBoost-based importance ranking, top 14 features kept
- `_train_and_log()` — generic train + MLflow log function used by all 4 baseline wrappers
- `_train_catboost_baseline()`, `_train_xgboost_baseline()`, `_train_lightgbm_baseline()`, `_train_gbm_baseline()`
- `_run_optuna_tuning()` — 50 TPE trials on CatBoost, each as a nested MLflow run; re-trains final model with best params
- `_register_best_model()` — registers lowest-RMSE run as `Diamond/Production` in MLflow Model Registry
- `_verify_model_loads()` — smoke test: loads registered model via `mlflow.pyfunc.load_model`
- `_log_common_tags()` — sets `model_type` and `python_version=3.11` on every run

### `src/tests/test_train.py` — 30 unit tests

All MLflow, model, and filesystem interactions mocked. No live server required.

Test classes: `TestComputeMetrics`, `TestGetFeatureNames`, `TestSelectFeatures`, `TestRunExists`, `TestGetOrCreateExperiment`, `TestTrainAndLog`, `TestRegisterBestModel`, `TestRunTraining`, `TestLogCommonTags`, `TestLoadData`, `TestBaselineTrainers`, `TestTrainAndLogExtended`, `TestRunOptunaTuning`, `TestVerifyModelLoads`.

---

## MLflow Results

| Run                | RMSE     | MAE    | R²     | MAPE |
| ------------------ | -------- | ------ | ------ | ---- |
| catboost-baseline  | 0.0969   | 0.0613 | 0.9674 | —    |
| xgboost-baseline   | 0.0952   | —      | 0.9685 | —    |
| lightgbm-baseline  | 0.0996   | —      | 0.9656 | —    |
| gbm-baseline       | 0.0988   | —      | 0.9661 | —    |
| **catboost-tuned** | **best** | —      | —      | —    |

> Note: metrics are in log-price space. R² > 0.96 on all baselines at default hyperparameters.

**Registered:** `Diamond v1` → `Production` stage in MLflow Model Registry.

**Artifacts per run:** `model/` (sklearn flavor + signature), `pipeline/pipeline.joblib`, selected feature names JSON.

---

## Issues Encountered and Resolved

### 1. `PermissionError: /mlruns` — artifact upload path

**Cause:** MLflow server was started with `--default-artifact-root /mlflow/artifacts` (a filesystem path inside the Docker container). When the local client created a run, it received `/mlflow/artifacts/...` as the artifact URI and tried to write to that path on the local machine — which doesn't exist and is not writable.

**Fix:** Changed `docker-compose.yml` to use the `mlflow-artifacts:/` URI scheme as the default artifact root, with `--artifacts-destination /mlflow/artifacts` for the server-side write location.

```yaml
--artifacts-destination /mlflow/artifacts
--default-artifact-root mlflow-artifacts:/
--serve-artifacts
```

The `mlflow-artifacts:/` scheme is special: the client automatically routes artifact uploads through `{MLFLOW_TRACKING_URI}/api/2.0/mlflow-artifacts/`, so the same code works from both local (`localhost:5000`) and Docker (`mlflow:5000`) without any host resolution issues.

### 2. `TypeError: MLflowCallback.__init__() got an unexpected keyword argument 'create_new_experiment'`

**Cause:** The `create_new_experiment` parameter was removed from `MLflowCallback` in the installed version of `optuna-integration`. The experiment was already created before this point anyway.

**Fix:** Removed `create_new_experiment=False` from the `MLflowCallback` constructor.

### 3. No model schema in MLflow UI

**Cause:** `mlflow.sklearn.log_model` was called without a `signature` argument.

**Fix:** Added `infer_signature(X_train, model.predict(X_train))` and passed the result as `signature=` to every `log_model` call in both `_train_and_log` and `_run_optuna_tuning`.

---

## Selected Features (top 14 of 22)

```text
Colour, Clarity, Polish, Symmetry, Fluorescence,
Shape_Pear, Shape_Round,
Weight, length, width, depth_mm,
volume, carat_per_volume, log_weight
```

Cut and Shape_* (except Pear and Round) were dropped by the CatBoost importance selector.

---

## Decisions Recorded

| Question         | Decision                                                        |
| ---------------- | --------------------------------------------------------------- |
| Confidence range | ±8% heuristic (applied at prediction time in API)               |
| Optuna budget    | 50 trials, TPESampler + MedianPruner                            |
| Artifact root    | `mlflow-artifacts:/` scheme — works from local and Docker       |
| Model signature  | Inferred from training data via `mlflow.models.infer_signature` |

---

## Day 3 Done Criteria — All Met

- [x] MLflow UI shows 5 runs (4 baselines + catboost-tuned) with metrics, params, and artifacts
- [x] Best model registered as `Diamond v1 / Production`
- [x] Model schema visible in MLflow UI (input: 14 double features, output: double)
- [x] `mlflow.pyfunc.load_model("models:/Diamond/Production")` verified on registration
- [x] R² > 0.96 on all baselines; tuned model is best
- [x] `make train` (Docker) and `uv run -m src.train` (local) both work
- [x] Skip-if-exists: re-running `make train` is safe and idempotent
- [x] `make test` passes with ≥ 80% coverage

---

## Up Next — Day 4: SHAP Explainability

- `src/explain.py` — `TreeExplainer` with 200-row background sample
- Global SHAP outputs: summary beeswarm, dependence plot (Weight), importance bar chart → logged to MLflow
- Local SHAP outputs: `get_shap_values()`, `get_waterfall_b64()` — for the API
- Explainer cached on startup, never recreated per request
- Teal for positive SHAP bars, coral for negative
