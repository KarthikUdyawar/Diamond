# 💎 Diamond — TODO

> Synthesized from: [PRD v1.0](./Diamond_PRD.md) · [Design Document v1.0](./Diamond_Design_Document.md) · [Tech Rules v1.0](./Diamond_Tech_Rules.md)  
> **Target:** 7 days · **Release:** v1.0  
> **Legend:** 🔴 Blocker · 🟡 Important · 🟢 Nice to have · ✅ Done

---

## 📅 Day 1 — Repo Restructure + Infrastructure Skeleton

### Repo Setup
- [ ] 🔴 Delete old Docker notebook setup (keep notebooks/ folder intact)
- [ ] 🔴 Create new folder structure: `src/`, `api/`, `ui/`, `data/raw/`, `data/processed/`, `docs/`
- [ ] 🔴 Move existing EDA notebook to `notebooks/eda.ipynb` (do not modify it)
- [ ] 🔴 Move `diamonds.csv` to `data/raw/diamonds.csv`
- [ ] 🔴 Create `pyproject.toml` with all pinned dependencies from Tech Rules §9
- [ ] 🔴 Create `.env.example` with all required variable names (no real values)
- [ ] 🔴 Create `.env` locally from `.env.example` (gitignored)
- [ ] 🔴 Update `.gitignore` — add `.env`, `data/processed/`, `mlruns/`, `__pycache__`, `.mypy_cache`, `.ruff_cache`
- [ ] 🟡 Create `src/__init__.py`, `api/__init__.py`, `ui/__init__.py`
- [ ] 🟡 Move `docs/` — place PRD, Design Doc, Tech Rules, and this TODO in `docs/`

### Docker Compose Skeleton
- [ ] 🔴 Write `docker-compose.yml` with 4 services: `postgres`, `mlflow`, `api`, `ui`
- [ ] 🔴 Use `postgres:15-alpine` image for the database service
- [ ] 🔴 Use `ghcr.io/mlflow/mlflow` for the MLflow service
- [ ] 🔴 Use `python:3.11-slim-bookworm` as base for `api` and `ui` custom images
- [ ] 🔴 Create single custom bridge network `diamond-net` — all services use it
- [ ] 🔴 Add health check to `postgres` service (pg_isready)
- [ ] 🔴 Add health check to `mlflow` service (HTTP GET /health)
- [ ] 🔴 Set startup order: `postgres` → (healthy) → `mlflow` → (healthy) → `api` → `ui`
- [ ] 🔴 Use named Docker volumes for PostgreSQL data and MLflow artifacts
- [ ] 🔴 Set restart policy: `unless-stopped` for postgres + mlflow; `on-failure` for api + ui
- [ ] 🟡 Expose only necessary ports: 5000 (mlflow), 8000 (api), 8501 (ui)
- [ ] 🟡 Pass all credentials via `.env` — no hardcoded values in `docker-compose.yml`

### Makefile
- [ ] 🔴 Create `Makefile` with self-documenting targets (use `##` comment convention)
- [ ] 🔴 Add `make up` — starts all Docker Compose services
- [ ] 🔴 Add `make down` — stops all services
- [ ] 🔴 Add `make mlflow` — opens `http://localhost:5000` in browser
- [ ] 🔴 Add `make train` — runs training pipeline inside container
- [ ] 🔴 Add `make features` — runs feature engineering only
- [ ] 🔴 Add `make test` — runs pytest with coverage report
- [ ] 🔴 Add `make ui` — opens `http://localhost:8501` in browser
- [ ] 🔴 Add `make clean` — removes containers, volumes, mlruns/
- [ ] 🟡 Add `make lint` — runs ruff + mypy
- [ ] 🟡 Add `make format` — runs ruff format
- [ ] 🟡 Add `make help` — prints all targets with descriptions

### Code Quality Tooling
- [ ] 🔴 Add `ruff` config to `pyproject.toml` (line-length 88, target py311, select E/F/I/N/UP/ANN/B/SIM)
- [ ] 🔴 Add `mypy` config to `pyproject.toml` (strict=true, ignore_missing_imports=true)
- [ ] 🔴 Create `.pre-commit-config.yaml` — runs ruff + mypy before every commit
- [ ] 🔴 Run `pre-commit install` locally
- [ ] 🟡 Verify `make lint` passes on empty scaffolding before proceeding to Day 2

### Day 1 Done Criteria
- [ ] `make up` starts postgres and mlflow without errors
- [ ] `make mlflow` opens the MLflow UI at `localhost:5000`
- [ ] Decide and document: should `mlruns/` be committed or gitignored? (PRD Q4)

---

## 📅 Day 2 — Feature Engineering Pipeline

### Data Validation
- [ ] 🔴 Create `src/features.py` — main feature engineering module
- [ ] 🔴 Add Pandera schema for raw CSV validation at pipeline start
- [ ] 🔴 Validate column names, dtypes, and value ranges on `diamonds.csv` before processing
- [ ] 🔴 Raise a clear error (not a pandas error) if schema validation fails

### Feature Transformations
- [ ] 🔴 Implement `OrdinalEncoder` for `cut` with exact order: `["Fair", "Good", "Very Good", "Premium", "Ideal"]`
- [ ] 🔴 Implement `OrdinalEncoder` for `color` with exact order: `["J", "I", "H", "G", "F", "E", "D"]`
- [ ] 🔴 Implement `OrdinalEncoder` for `clarity` with exact order: `["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]`
- [ ] 🔴 Implement log1p transform for `carat` using `FunctionTransformer(np.log1p)`
- [ ] 🔴 Engineer `volume = x * y * z` feature via custom `FunctionTransformer`
- [ ] 🔴 Engineer `carat_per_volume` ratio feature via custom `FunctionTransformer`
- [ ] 🔴 Keep `depth` and `table` as-is (no transformation)
- [ ] 🔴 Remove `price` from features (it is the target — log-transform it separately for the target vector)

### Pipeline Assembly
- [ ] 🔴 Wrap all transformations in a single `sklearn.pipeline.Pipeline` object
- [ ] 🔴 Use `ColumnTransformer` to apply different transforms to different columns
- [ ] 🔴 Ensure the pipeline is fit **only on training data** — never the full dataset
- [ ] 🔴 Add `train_test_split(test_size=0.2, random_state=42)` — used by all subsequent steps
- [ ] 🔴 Save fitted pipeline as `pipeline.joblib` using `joblib.dump`
- [ ] 🔴 Write a `load_pipeline()` function that loads from MLflow artifact store (not local path)
- [ ] 🟡 Save processed Parquet files to `data/processed/train.parquet` and `data/processed/test.parquet`

### Type Hints & Tests
- [ ] 🔴 Add type hints to all functions in `src/features.py`
- [ ] 🔴 Write `src/tests/test_features.py` — test pipeline fit/transform produces expected shape and dtypes
- [ ] 🟡 Verify no data leakage: test set rows must not appear in pipeline fitting

### Day 2 Done Criteria
- [ ] `make features` runs without error and produces Parquet files in `data/processed/`
- [ ] Pipeline object serializes and deserializes without data loss
- [ ] `make lint` still passes

---

## 📅 Day 3 — Model Training + MLflow + Optuna

### MLflow Experiment Setup
- [ ] 🔴 Create `src/train.py` — main training script
- [ ] 🔴 Set MLflow tracking URI from env var `MLFLOW_TRACKING_URI`
- [ ] 🔴 Create or get experiment named `diamond-price` (from env var `MLFLOW_EXPERIMENT_NAME`)
- [ ] 🔴 Decide: confidence range from quantile regression or ±8% heuristic? (PRD Q1, default: ±8%)
- [ ] 🔴 Decide: Optuna cap at 50 trials or 10 minutes? (PRD Q2, default: 50 trials)

### Train All 4 Models
- [ ] 🔴 Train `XGBoostRegressor` — log as MLflow run named `xgboost-baseline`
- [ ] 🔴 Train `LGBMRegressor` — log as MLflow run named `lightgbm-baseline`
- [ ] 🔴 Train `CatBoostRegressor` — log as MLflow run named `catboost-baseline`
- [ ] 🔴 Train `GradientBoostingRegressor` — log as MLflow run named `gbm-baseline`

### MLflow Logging (mandatory for every run)
- [ ] 🔴 Log all hyperparameters via `mlflow.log_params(model.get_params())`
- [ ] 🔴 Log metrics on test set: RMSE, MAE, R², MAPE via `mlflow.log_metrics()`
- [ ] 🔴 Log model artifact via `mlflow.sklearn.log_model(model, "model")`
- [ ] 🔴 Log fitted pipeline as artifact: `mlflow.log_artifact("pipeline.joblib")`
- [ ] 🔴 Set tag `model_type` (e.g. `"xgboost"`)
- [ ] 🔴 Set tag `python_version` = `"3.11"`
- [ ] 🟡 Log training time as a metric: `mlflow.log_metric("training_time_sec", elapsed)`

### Optuna Hyperparameter Tuning
- [ ] 🔴 Run 50 Optuna trials on the best baseline model (expected: XGBoost)
- [ ] 🔴 Use `TPESampler` (default) and `MedianPruner`
- [ ] 🔴 Log each trial as a nested MLflow run via `MLflowCallback`
- [ ] 🔴 Use the XGBoost search space from Tech Rules §4.3 exactly
- [ ] 🔴 Re-train final model with best trial params and log as a new MLflow run `xgboost-tuned`

### Model Registry
- [ ] 🔴 Register the best model (lowest RMSE) in MLflow Model Registry as `Diamond`
- [ ] 🔴 Promote it to `Production` stage: `models:/Diamond/Production`
- [ ] 🔴 Verify the API can load it via `mlflow.pyfunc.load_model("models:/Diamond/Production")`

### Day 3 Done Criteria
- [ ] MLflow UI shows at least 4+ runs with metrics, params, and artifacts
- [ ] Best model is registered under `Diamond/Production` in the registry
- [ ] R² > 0.98 and RMSE < $550 on the test set (if not, tune further before proceeding)

---

## 📅 Day 4 — SHAP Explainability

### SHAP Module Setup
- [ ] 🔴 Create `src/explain.py` — SHAP logic module
- [ ] 🔴 Instantiate `shap.TreeExplainer(model)` with a background dataset of 200 randomly sampled training rows (`random_state=42`)
- [ ] 🔴 Add type hints to all functions

### Global SHAP Outputs (per training run)
- [ ] 🔴 Generate SHAP summary beeswarm plot using `shap.summary_plot` on the test set
- [ ] 🔴 Save as `shap_summary.png` and log to MLflow via `mlflow.log_artifact("shap_summary.png")`
- [ ] 🔴 Generate SHAP dependence plot for `carat` using `shap.dependence_plot`
- [ ] 🔴 Save as `shap_dependence_carat.png` and log to MLflow as artifact
- [ ] 🟡 Generate feature importance bar chart (mean absolute SHAP) and log as `shap_importance.png`

### Local SHAP Outputs (per prediction — for API)
- [ ] 🔴 Write `get_shap_values(input_row: pd.DataFrame) -> dict` — returns per-feature SHAP values
- [ ] 🔴 Write `get_waterfall_b64(input_row: pd.DataFrame) -> str` — returns base64-encoded PNG of waterfall plot
- [ ] 🔴 Ensure the `TreeExplainer` object is instantiated **once** and cached — never recreated per request
- [ ] 🔴 Use teal for positive SHAP bars, coral for negative (matches Design Doc §9 colour guidance)
- [ ] 🔴 Add feature labels and dollar signs to all SHAP chart axes

### Integration with Training
- [ ] 🔴 Call `src/explain.py` functions at the end of each training run in `src/train.py`
- [ ] 🔴 Verify SHAP plots appear as artifacts in the MLflow UI for all 4+ runs

### Day 4 Done Criteria
- [ ] Every MLflow run has `shap_summary.png` and `shap_dependence_carat.png` as artifacts
- [ ] `get_waterfall_b64()` returns a valid base64 PNG for a sample input
- [ ] SHAP values are computed in < 200ms for a single input row

---

## 📅 Day 5 — FastAPI REST API

### API Structure
- [ ] 🔴 Create `api/main.py` — FastAPI app factory with `lifespan` context manager
- [ ] 🔴 Create `api/schemas.py` — all Pydantic v2 request/response models
- [ ] 🔴 Create `api/routes/predict.py` — `POST /predict` route
- [ ] 🔴 Create `api/routes/explain.py` — `POST /explain` route
- [ ] 🔴 Create `api/routes/health.py` — `GET /health` route
- [ ] 🔴 Create `api/services/model.py` — model loading and inference logic
- [ ] 🔴 Create `api/services/shap.py` — SHAP explainer wrapper for API use

### Model Loading
- [ ] 🔴 Load model at startup via `lifespan`: `mlflow.pyfunc.load_model("models:/Diamond/Production")`
- [ ] 🔴 Load fitted pipeline from MLflow artifact at startup
- [ ] 🔴 Load and cache `TreeExplainer` at startup — never recreated per request
- [ ] 🔴 Decide: Streamlit calls FastAPI directly or shares inference code? (Design Q3, default: call API directly)
- [ ] 🟡 Log model version and run ID on startup: `"Loaded Diamond v1 (run: abc123)"`

### Pydantic v2 Input Schema
- [ ] 🔴 Implement `DiamondInput` with `model_config = {"strict": True}`
- [ ] 🔴 `carat`: float, ge=0.2, le=5.01
- [ ] 🔴 `cut`: `Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]`
- [ ] 🔴 `color`: `Literal["D", "E", "F", "G", "H", "I", "J"]`
- [ ] 🔴 `clarity`: `Literal["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]`
- [ ] 🔴 `depth`: float, ge=43.0, le=79.0
- [ ] 🔴 `table`: float, ge=43.0, le=95.0
- [ ] 🔴 `x`, `y`, `z`: float, ge=0.0, le=31.8
- [ ] 🔴 Implement `PredictResponse` with `predicted_price_usd`, `confidence_range` (low/high), `model_version`, `mlflow_run_id`
- [ ] 🔴 Implement `ExplainResponse` with `shap_values` dict, `base_value`, `waterfall_plot_b64`
- [ ] 🔴 Implement `HealthResponse` with `status`, `model_name`, `model_version`, `mlflow_run_id`

### Endpoints
- [ ] 🔴 `POST /predict` — transform input → run inference → return `PredictResponse`
- [ ] 🔴 `POST /explain` — transform input → compute SHAP → return `ExplainResponse`
- [ ] 🔴 `GET /health` — return model metadata from cached model loader
- [ ] 🔴 Enable CORS for `localhost:8501` only
- [ ] 🔴 Return HTTP 422 for validation errors (Pydantic default — verify it works)
- [ ] 🔴 Return HTTP 500 with sanitised message for unhandled exceptions (no stack traces in response)
- [ ] 🟡 Log prediction latency on every `/predict` call: `"Prediction completed in 42ms"`
- [ ] 🟡 Log at INFO level: input received, model used, response sent

### API Rules Compliance
- [ ] 🔴 No `print()` statements anywhere in `api/` — use `logging` module only
- [ ] 🔴 Route handlers contain zero business logic — call service functions only
- [ ] 🔴 All service functions are pure (no side effects except logging)
- [ ] 🔴 `/predict` responds in < 100ms (p95) — test manually with `curl` timing

### API Tests
- [ ] 🔴 Create `api/tests/conftest.py` — shared fixtures including mocked model and SHAP explainer
- [ ] 🔴 Create `api/tests/test_predict.py`:
  - [ ] `test_predict_returns_price_for_valid_input`
  - [ ] `test_predict_rejects_invalid_carat`
  - [ ] `test_predict_rejects_invalid_cut`
  - [ ] `test_predict_returns_422_for_missing_field`
- [ ] 🔴 Create `api/tests/test_explain.py`:
  - [ ] `test_explain_returns_shap_values_for_valid_input`
  - [ ] `test_explain_returns_base64_png`
  - [ ] `test_explain_shap_values_sum_to_prediction_minus_base`
- [ ] 🔴 Create `api/tests/test_health.py`:
  - [ ] `test_health_returns_ok_status`
  - [ ] `test_health_returns_model_version`
- [ ] 🔴 Achieve ≥ 80% test coverage on `api/`
- [ ] 🟡 Create `api/tests/fixtures.py` — fixed seed input for all tests: `{carat: 0.89, cut: "Premium", ...}`

### Day 5 Done Criteria
- [ ] `curl -X POST http://localhost:8000/predict -d '{...}'` returns correct JSON in < 100ms
- [ ] `curl -X POST http://localhost:8000/explain` returns SHAP values and a base64 PNG
- [ ] `make test` passes with ≥ 80% coverage

---

## 📅 Day 6 — Streamlit Dashboard

### App Structure
- [ ] 🔴 Create `ui/app.py` — main Streamlit app entry point
- [ ] 🔴 Create `ui/api_client.py` — all httpx calls to FastAPI go through here only
- [ ] 🔴 Create `ui/constants.py` — default input values and template strings (never hardcode in page files)
- [ ] 🔴 Create `ui/templates.py` — plain-English sentence template functions

### Default Input Values (in `constants.py`)
- [ ] 🔴 Set default "average diamond" pre-filled values: `carat=0.89, cut="Premium", color="H", clarity="SI2", depth=62.4, table=58.0, x=6.15, y=6.10, z=3.83`
- [ ] 🔴 Initialise `st.session_state` with defaults at app startup so output panel is never empty on first load

### State Management
- [ ] 🔴 Store `inputs` in `st.session_state` — shared across Tab 1 and Tab 2
- [ ] 🔴 Store `prediction` result in `st.session_state` — updated on Predict button click
- [ ] 🔴 Store `mlflow_runs` in `st.session_state` — fetched on Tab 3 open, refreshed every 60s
- [ ] 🔴 No global variables — `st.session_state` is the only state store

### Tab 1 — Price Predictor
- [ ] 🔴 Two-column layout: Input Panel (left), Output Panel (right)
- [ ] 🔴 Input Panel — Sliders: `carat`, `depth`, `table`, `x`, `y`, `z`
- [ ] 🔴 Input Panel — Dropdowns: `cut`, `color`, `clarity`
- [ ] 🔴 Sliders fire prediction on mouse release (not on drag)
- [ ] 🔴 Dropdowns fire prediction instantly on change
- [ ] 🔴 Output Panel — `PriceBadge`: predicted price formatted as `$3,842` (large, monospaced font)
- [ ] 🔴 Output Panel — `ConfidenceBar`: horizontal bar showing low–high range, colour shifts green → amber based on range width
- [ ] 🔴 Output Panel — Plain-English summary sentence from template (top SHAP feature drives the sentence)
- [ ] 🔴 Output Panel — Top 3 SHAP contributors: `↑ carat +$1,243`, `↓ color -$122` format
- [ ] 🔴 Output Panel — Model metadata footer: `"Model: XGBoost v1 · R² 0.984"` in small grey text
- [ ] 🔴 Error state — API unreachable: orange banner `"Prediction service is offline. Run make up to start."`
- [ ] 🟡 Error state — Prediction > 300ms: subtle pulsing animation on output card (no spinner)
- [ ] 🟡 Error state — API error: `"Something went wrong. Check the API logs."` card replaces output

### Tab 2 — Explain This Prediction
- [ ] 🔴 Inputs synced from Tab 1 (same `st.session_state.inputs`)
- [ ] 🔴 Summary line at top: `"Explaining: 0.89ct · Premium · H · SI2"`
- [ ] 🔴 Show `"Predicted Price: $3,842   Base Value: $3,812"`
- [ ] 🔴 Render SHAP waterfall chart using Plotly (from base64 or re-rendered with Plotly directly)
- [ ] 🔴 Waterfall bars: teal for positive, coral for negative (not red/green)
- [ ] 🔴 Feature contribution table: Feature | Value | Impact (+$/-$) | Plain English columns
- [ ] 🔴 Sort table by absolute SHAP value descending
- [ ] 🔴 Info tooltip on "Base Value": explains what the base value means in plain English
- [ ] 🔴 Auto-explain on Tab 2 load using Tab 1's current inputs (Design Q U1 resolved)
- [ ] 🟡 "Back to Predictor" button that switches to Tab 1

### Tab 3 — Model Dashboard
- [ ] 🔴 Active Model Card at top: model name, version, run ID, R², training date
- [ ] 🔴 Highlight active model card with accent colour border
- [ ] 🔴 Fetch run data live from MLflow API: `GET http://localhost:5000/api/2.0/mlflow/runs/search`
- [ ] 🔴 Show "Last refreshed: [timestamp]" — auto-refresh every 60 seconds
- [ ] 🔴 Run comparison table: Model | Run ID | R² | RMSE | MAE | Status columns
- [ ] 🔴 Active model row highlighted with ✅ badge
- [ ] 🔴 RMSE bar chart (Plotly): sorted ascending, active model bar in accent colour
- [ ] 🔴 Global SHAP importance horizontal bar chart: pulled from MLflow artifact store of active run
- [ ] 🔴 "Open MLflow UI" button — opens `http://localhost:5000` in new tab

### Visual Direction (Design Doc §9)
- [ ] 🟡 Override default Streamlit theme — dark neutral background (deep slate or charcoal)
- [ ] 🟡 Surface cards slightly lighter than background
- [ ] 🟡 Single accent colour (teal or electric blue) used sparingly
- [ ] 🟡 Hero price number uses monospaced / tabular-figures font
- [ ] 🟡 Avoid default Streamlit grey sidebar + white background look
- [ ] 🟡 Apply custom CSS via `st.markdown` with `unsafe_allow_html=True` for theme overrides

### Accessibility
- [ ] 🟡 All sliders and dropdowns have descriptive labels (not just field names)
- [ ] 🟡 SHAP colours use teal/coral not red/green (colour-blind safe)
- [ ] 🟡 Error messages include icon (⚠️) and text, not colour alone
- [ ] 🟡 Minimum 14px body text; price hero ≥ 40px

### Day 6 Done Criteria
- [ ] All 3 tabs render without errors at `localhost:8501`
- [ ] Predict and Explain calls succeed end-to-end from UI
- [ ] Tab 3 shows live MLflow run data
- [ ] Dashboard does not look like a default Streamlit template

---

## 📅 Day 7 — CI, Polish, and v1.0 Release

### GitHub Actions CI
- [ ] 🔴 Create `.github/workflows/ci.yml`
- [ ] 🔴 Trigger on push and PR to `main`
- [ ] 🔴 Step 1: Checkout
- [ ] 🔴 Step 2: Set up Python 3.11
- [ ] 🔴 Step 3: `pip install -e ".[dev]"`
- [ ] 🔴 Step 4: Run `ruff check src/ api/ ui/`
- [ ] 🔴 Step 5: Run `mypy src/ api/ ui/`
- [ ] 🔴 Step 6: Run `pytest api/tests/ --cov=api --cov-fail-under=80`
- [ ] 🔴 Step 7: Upload coverage report to job summary
- [ ] 🔴 Protect `main` branch — require CI to pass before merge, no direct pushes

### README Overhaul
- [ ] 🔴 Write new `README.md` — replace old notebook README entirely
- [ ] 🔴 Add project title + one-line description
- [ ] 🔴 Add architecture diagram (ASCII or image)
- [ ] 🔴 Add prerequisites section: Docker, Docker Compose V2, make
- [ ] 🔴 Add quickstart: `git clone` → `cp .env.example .env` → `make up` → `make train` → `make ui`
- [ ] 🔴 Add screenshots of MLflow UI and Streamlit dashboard (all 3 tabs)
- [ ] 🔴 Add API reference section: `POST /predict`, `POST /explain`, `GET /health` with example curl commands
- [ ] 🟡 Add "How it works" section explaining the ML pipeline (feature engineering → training → SHAP → serving)
- [ ] 🟡 Add badges: CI status, Python version, License, MLflow

### Final Polish
- [ ] 🔴 Run `make lint` — ensure ruff + mypy pass with zero errors
- [ ] 🔴 Run `make test` — ensure ≥ 80% coverage
- [ ] 🔴 Run `make clean` then `make up` from a clean state — verify cold start < 60 seconds
- [ ] 🔴 Run `make train` — verify 4+ runs appear in MLflow, best model registered
- [ ] 🔴 Open `make ui` — manually test all 3 tabs end-to-end
- [ ] 🟡 Test on a different machine or fresh Docker environment if possible

### v1.0 Release Checklist (from PRD §10)
- [ ] 🔴 R² > 0.98 on held-out test set
- [ ] 🔴 MLflow UI shows at least 4 model runs with metrics and SHAP artifacts
- [ ] 🔴 Best model registered in MLflow Model Registry under `Diamond/Production`
- [ ] 🔴 `/predict` endpoint returns correct predictions with Pydantic validation
- [ ] 🔴 `/explain` endpoint returns SHAP values and a base64 waterfall plot
- [ ] 🔴 Streamlit dashboard renders all 3 tabs without errors
- [ ] 🔴 `make up` starts the full stack from a clean clone
- [ ] 🔴 GitHub Actions CI passes on `main` branch
- [ ] 🔴 README includes architecture diagram, feature list, and setup instructions
- [ ] 🔴 All secrets handled via `.env` — no hardcoded credentials anywhere
- [ ] 🔴 Tag release as `v1.0` on GitHub

---

## 🔖 Decisions to Make Before Starting

These are the open questions from the three docs — resolve them before Day 3:

| #   | Question                                                   | Default               | Decide By          |
| --- | ---------------------------------------------------------- | --------------------- | ------------------ |
| Q1  | Confidence range: quantile regression or ±8% heuristic?    | ±8% heuristic         | Day 3              |
| Q2  | Optuna: 50 trial cap or 10-minute time box?                | 50 trials             | Day 3              |
| Q3  | Streamlit calls FastAPI directly or shares inference code? | Call FastAPI directly | Day 5              |
| Q4  | `mlruns/` committed to repo or gitignored?                 | Gitignored            | Day 1              |
| U1  | Tab 2 auto-explain on load or wait for button click?       | Auto-explain          | Day 6              |
| U2  | Confidence range source?                                   | ±8% heuristic         | Day 3 (same as Q1) |
| U5  | Export Explanation PDF in v1 or deferred?                  | Defer to v1.1         | Day 6              |

---

## 🚫 Out of Scope — Do Not Build in v1.0

Do not let these creep in. If tempted, add them to a `v2-ideas.md` instead:

- Real-time streaming data ingestion
- User authentication / multi-user support
- Cloud deployment (AWS / GCP / Azure)
- Mobile UI
- A/B model testing in production
- Dark / light theme toggle
- Multi-diamond batch comparison
- Animated diamond 3D model
- Export Explanation PDF
- Neural network models (TabNet, MLP)
- Redis caching layer
- Kubernetes orchestration
- Alembic migrations
- Celery task queue

---

## 📊 Progress Tracker

| Day | Area                                         | Status        | Blocker?   |
| --- | -------------------------------------------- | ------------- | ---------- |
| 1   | Repo restructure + Docker Compose + Makefile | ⬜ Not started | —          |
| 2   | Feature engineering pipeline                 | ⬜ Not started | Day 1 done |
| 3   | Model training + MLflow + Optuna             | ⬜ Not started | Day 2 done |
| 4   | SHAP explainability                          | ⬜ Not started | Day 3 done |
| 5   | FastAPI REST API + tests                     | ⬜ Not started | Day 4 done |
| 6   | Streamlit dashboard                          | ⬜ Not started | Day 5 done |
| 7   | CI + README + v1.0 release                   | ⬜ Not started | Day 6 done |

> Update status to: ⬜ Not started · 🔄 In progress · ✅ Done · 🚫 Blocked

---

*TODO v1.0 — Diamond Project — March 2026*  
*Generated from PRD, Design Document, and Tech Rules. All items traceable to source docs.*