# 💎 Diamond — Day 4 Complete

> **SHAP Explainability + Structured Logging + Config + API Move**
> Completed: March 2026

---

## What Was Built

### `src/config.py` — Centralised Configuration

Single `Settings` class built on Pydantic `BaseSettings`. All configurable values in one place — no more hardcoded strings scattered across modules.

- `get_settings()` cached via `@lru_cache` — one `Settings` instance per process lifetime
- Environment switching is **explicit**: caller passes `--env-file .env.local` to uv or `env_file: .env.staging` in Docker Compose — no silent file-precedence surprises
- Full validation on startup: invalid `APP_ENV`, out-of-range `SHAP_BACKGROUND_ROWS`, bad `TRAIN_TEST_SPLIT_SIZE` all raise `ValidationError` immediately
- Derived properties: `postgres_dsn`, `mlflow_model_uri`, `pipeline_path`, `train_parquet_path`, `test_parquet_path`, `is_local`, `is_staging`, `is_testing`
- `T201` added to ruff ignore list for the intentional `print()` in `_warn_empty_secrets` (logger not yet initialised at `Settings` construction time)

### `src/logger.py` — Structured Logging

`structlog`-based logger replacing all `logging.getLogger(__name__)` calls across the codebase.

- `get_logger(name)` — public API, configures on first call, cheap on subsequent calls
- `bind_context(**kwargs)` / `clear_context()` — thread-local key-value binding, included in all subsequent log calls without explicit passing
- **local / testing** → `ConsoleRenderer` (coloured + human-readable; colours disabled in pytest)
- **staging** → `JSONRenderer` (one JSON object per line, machine-readable)
- Log level driven by `settings.log_level`: `DEBUG` local, `WARNING` testing, `INFO` staging
- Noisy third-party loggers silenced to `WARNING`: `urllib3`, `httpx`, `mlflow`, `optuna`, `catboost`, `LightGBM`
- `configure_logging()` is idempotent — safe to call multiple times

### `src/explain.py` — SHAP Explainability

`ExplainerService` class wrapping `shap.TreeExplainer` with two output modes:

**Global (per training run — logged to MLflow):**

- `plot_summary(X)` → `shap_summary.png` (beeswarm)
- `plot_dependence(X, feature="Weight")` → `shap_dependence_weight.png`
- `plot_importance(X)` → `shap_importance.png` (mean |SHAP| bar chart)

**Local (per prediction — served by the API):**

- `get_shap_values(input_row)` → `dict[feature_name, shap_value]`, sorted by absolute value descending
- `get_waterfall_b64(input_row)` → base64-encoded PNG waterfall chart

**Design decisions:**

- `ExplainerService` instantiated once and cached at module level via `init_explainer()` / `get_explainer()`
- Background sample: `shap_background_rows` rows (default 200), `shap_random_state` seed (default 42) — both from config
- Colours: teal `#2a9d8f` positive, coral `#e76f51` negative — consistent across all plots
- `matplotlib` backend set to `"Agg"` (non-interactive, safe in headless Docker)
- All figures closed immediately after saving to prevent memory leaks in long-running API processes
- `clear_explainer()` available for test teardown

### `src/features.py` — Updated

- All `logging.getLogger(__name__)` → `get_logger(__name__)`
- All `logging.info(...)` string-format calls → structured `logger.info("event_name", key=value)`
- All hardcoded path constants (`RAW_DATA_DIR`, `PROCESSED_DIR`, `PIPELINE_PATH`) → `get_settings()` derived paths
- Caller-supplied `raw_dir` / `processed_dir` / `pipeline_path` kwargs still override config — existing test suite passes unchanged
- `bind_context` / `clear_context` wrapping `run_feature_engineering()`

### `src/train.py` — Updated

- `load_dotenv(override=True)` + `os.environ.get()` → `get_settings()`
- All `logging.info(...)` → structured `get_logger(__name__)` calls
- `bind_context(app_env, experiment)` at entry, `clear_context()` at exit
- SHAP artifacts integrated: `_log_shap_artifacts(final_model, X_train, selected_names)` called inside the tuning parent run context (best model only — tuned CatBoost)
- All existing logic preserved exactly: `run_scope` provenance system, idempotency guards, skip-if-exists, `_is_already_registered` check

### `src/api/__init__.py` — Scaffold

Empty package scaffold. Full implementation in Day 5.

### `dockerfiles/api.Dockerfile` — Moved

Moved from `api/Dockerfile` to `dockerfiles/api.Dockerfile`. Build context unchanged (project root). `docker-compose.yml` updated to reference new path.

### Multi-Environment Setup

Three env files replacing the single `.env`:

| File           | `APP_ENV` | `LOG_LEVEL` | `MLFLOW_TRACKING_URI`   | `SHAP_BACKGROUND_ROWS` | `OPTUNA_N_TRIALS` |
| -------------- | --------- | ----------- | ----------------------- | ---------------------- | ----------------- |
| `.env.local`   | `local`   | `DEBUG`     | `http://localhost:5000` | 200                    | 50                |
| `.env.testing` | `testing` | `WARNING`   | `http://localhost:5000` | 10                     | 2                 |
| `.env.staging` | `staging` | `INFO`      | `http://mlflow:5000`    | 200                    | 50                |

`.env` kept as alias for `.env.local`. All three files gitignored. Only `.env.example` committed.

**Switching mechanism:** `--env-file` flag passed explicitly:

```bash
uv run --env-file .env.local    -m src.features
uv run --env-file .env.testing  -m pytest
# Docker Compose picks up .env.staging via env_file: in docker-compose.yml
```

---

## Test Results

```text
145 passed, 19 warnings in 38.74s

Name                  Stmts   Miss  Cover
-------------------------------------------
src/config.py            65      4    94%
src/explain.py          128      7    95%
src/features.py         150     10    93%
src/logger.py            60      8    87%
src/train.py            287     28    90%
src/constants.py         55      0   100%
src/__init__.py           0      0   100%
src/api/__init__.py       0      0   100%
-------------------------------------------
TOTAL                   745     57    92%

Required test coverage of 80% reached. Total coverage: 92.35%
```

### New test files

| File                        | Tests | What's covered                                                                                                                                                     |
| --------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/tests/test_config.py`  | 24    | Defaults, env overrides, validation errors, derived properties, `get_settings()` cache                                                                             |
| `src/tests/test_logger.py`  | 16    | `get_logger()`, `configure_logging()` idempotency, level setting, noisy logger silencing, context binding, all renderers                                           |
| `src/tests/test_explain.py` | 20    | `ExplainerService` init, `get_shap_values()` dict shape/sorting/types, `get_waterfall_b64()` base64 validity, all three global plots, module-level cache lifecycle |

### Updated test file

`src/tests/test_train.py` — all 60 existing tests preserved unchanged. Added `patch("src.train._log_shap_artifacts")` to every `TestRunOptunaTuning` test (prevents real SHAP init against mocked models). Added one new test `test_shap_artifacts_called_for_tuned_model` verifying `_log_shap_artifacts` is called exactly once per tuning run.

---

## Issues Encountered and Resolved

### 1. `test_colors_from_config` — asserting `'' == '#2a9d8f'`

**Cause:** The `explainer` fixture set env vars via `monkeypatch` but did not call `get_settings.cache_clear()` afterwards. A stale `Settings` instance from a previous test (which hadn't set `SHAP_POSITIVE_COLOR`) was returned, giving empty strings.

**Fix:** Added explicit `monkeypatch.setenv` for both color vars and `get_settings.cache_clear()` inside the `explainer` fixture, forcing a fresh `Settings` read with the correct env in place.

### 2. `test_plot_importance_creates_png` — `ValueError: Invalid RGBA argument: ''`

**Cause:** Same stale cache issue — `_pos_color == ""` was passed to `matplotlib` when drawing the importance bar chart, which rejected the empty string as an invalid color.

**Fix:** Resolved by the same `explainer` fixture fix above.

### 3. `test_init_samples_background_rows` — `ValidationError: shap_background_rows >= 10`

**Cause:** Test used `SHAP_BACKGROUND_ROWS=5`, which correctly fails Pydantic's `ge=10` validator. The validator was working as intended — the test value was wrong.

**Fix:** Changed to `SHAP_BACKGROUND_ROWS=15` (valid, and less than the 20-row `sample_df`) so the sampling cap is still exercised meaningfully.

---

## Decisions Recorded

| Question              | Decision                                                         |
| --------------------- | ---------------------------------------------------------------- |
| Environment switching | `--env-file` passed explicitly — no silent `.env` auto-detection |
| `.env` file           | Kept as alias for `.env.local`                                   |
| Logging library       | `structlog` — JSON in staging, pretty in local/testing           |
| SHAP plots            | Best model only (tuned CatBoost) — not all 4 baselines           |
| `api/` location       | Moved to `src/api/` — full implementation in Day 5               |
| Dockerfile location   | `dockerfiles/api.Dockerfile`                                     |
| API tests location    | `src/tests/api/` — merged into single test tree                  |

---

## Day 4 Done Criteria — All Met

- [x] `make features` and `make train` use config + structured logs — no bare `print()` or `logging.getLogger`
- [x] `APP_ENV=staging` produces JSON logs; `APP_ENV=local` produces pretty coloured logs
- [x] MLflow tuned-CatBoost run has `shap/shap_summary.png`, `shap/shap_dependence_weight.png`, `shap/shap_importance.png` as artifacts
- [x] `get_waterfall_b64()` returns valid base64 PNG
- [x] `make test --env-file .env.testing` passes — 145 tests, 92.35% coverage (≥ 80% required)
- [x] `api/` fully removed; `src/api/__init__.py` scaffold in place; `dockerfiles/api.Dockerfile` created
- [x] `.env.local`, `.env.testing`, `.env.staging`, `.env.example` in place
- [x] `docker-compose.yml` updated: `dockerfile: dockerfiles/api.Dockerfile`, `env_file: .env.staging`
- [x] No hardcoded credentials or paths anywhere in `src/`

---

## Up Next — Day 5: FastAPI REST API

- `src/api/main.py` — FastAPI app with `lifespan` context manager (loads model + explainer on startup)
- `src/api/schemas.py` — Pydantic v2 `DiamondInput`, `PredictResponse`, `ExplainResponse`, `HealthResponse`
- `src/api/routes/predict.py` — `POST /predict`
- `src/api/routes/explain.py` — `POST /explain`
- `src/api/routes/health.py` — `GET /health`
- `src/api/services/model.py` — model loading and inference
- `src/api/services/shap.py` — SHAP explainer wrapper
- `src/tests/api/` — conftest, test_predict, test_explain, test_health
- CORS for `localhost:8501` only
- `/predict` < 100ms p95
- ≥ 80% test coverage on `src/api/`
