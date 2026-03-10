# 💎 Diamond — Tech Rules (Tech Stack Definition)

**Version:** 1.0  
**Author:** Karthik Udyawar  
**Companion Docs:** [PRD](./Diamond_PRD.md) · [Design Document](./Diamond_Design_Document.md)  
**Status:** Approved  
**Last Updated:** March 2026  

---

## 0. Purpose of This Document

This document is the **single source of truth** for every technology decision in the Diamond project. It answers three questions for every layer of the stack:

1. **What** tool/library/version are we using?
2. **Why** this one and not the obvious alternative?
3. **What are the rules** for how it must be used?

Any deviation from these rules requires a documented justification in the PR description. "It worked on my machine" is not a justification.

---

## 1. Non-Negotiable Global Rules

These apply everywhere, no exceptions:

| Rule                              | Detail                                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Python 3.11 only**              | No 3.9, no 3.12. 3.11 is the sweet spot — stable, fast, supported by all libs in this stack    |
| **No hardcoded secrets**          | All credentials go in `.env`. `.env` is in `.gitignore`. Period.                               |
| **Docker-first**                  | Every service runs in a container. "Works locally without Docker" is not a goal                |
| **Pinned dependencies**           | Every package has an exact version in `pyproject.toml`. No `>=`, no `~=`, no `*`               |
| **One Makefile to rule them all** | Every operation has a `make` target. No one should need to remember a raw `docker run` command |
| **Type hints everywhere**         | All function signatures must have type hints. `mypy` will enforce this in CI                   |
| **No Jupyter in production code** | Notebooks live in `notebooks/` and are for EDA only. No business logic in `.ipynb` files       |

---

## 2. Runtime & Environment

### 2.1 Python

| Item                 | Decision                                                           |
| -------------------- | ------------------------------------------------------------------ |
| Version              | **3.11.x** (latest patch)                                          |
| Base Docker image    | `python:3.11-slim-bookworm`                                        |
| Package manager      | **pip** with `pyproject.toml` (PEP 517/518 compliant)              |
| Dependency locking   | `pip-compile` → `requirements.txt` generated from `pyproject.toml` |
| Virtual environments | Inside Docker only. No local venvs committed                       |

**Why not conda?** Conda is too heavy for Docker and introduces environment resolution conflicts with Docker layer caching. pip + pinned versions inside Docker is simpler and reproducible.

**Why not Poetry?** Poetry is excellent but adds complexity with its own lock format and resolver. For this project size, pip-compile gives the same guarantees with less overhead.

### 2.2 Environment Variables

All secrets and config live in `.env` at the project root:

```bash
# .env (never commit this file)
POSTGRES_USER=diamond
POSTGRES_PASSWORD=diamond_secret
POSTGRES_DB=diamond_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=diamond-price

API_HOST=0.0.0.0
API_PORT=8000

STREAMLIT_PORT=8501
```

`.env.example` (committed, no real values) serves as the canonical reference for required variables.

---

## 3. Data Layer

### 3.1 Database

| Item               | Decision                                                   |
| ------------------ | ---------------------------------------------------------- |
| Engine             | **PostgreSQL 15**                                          |
| Docker image       | `postgres:15-alpine`                                       |
| ORM                | **SQLAlchemy 2.x** (Core only, not ORM for MLflow backend) |
| Migrations         | Not required for v1.0 (MLflow manages its own schema)      |
| Connection pooling | SQLAlchemy default pool (5 connections)                    |

**Why PostgreSQL and not SQLite?**  
MLflow's backend store supports SQLite but it cannot handle concurrent writes — if training and the API both try to log simultaneously, SQLite corrupts. PostgreSQL handles this correctly. You already have PostgreSQL experience from chronos and ledgermind, so there is zero learning curve.

**Rules:**
- Connection strings must be assembled from env vars only, never hardcoded
- All DB operations use context managers (`with engine.connect() as conn`)
- The `postgres` service must have a health check before MLflow starts (use `depends_on` with `condition: service_healthy`)

### 3.2 Data Files

| Item            | Decision                                                                |
| --------------- | ----------------------------------------------------------------------- |
| Raw data        | `data/raw/diamonds.csv` — committed to repo (54K rows, 3MB, acceptable) |
| Processed data  | `data/processed/` — gitignored, regenerated by `make features`          |
| Format          | CSV for raw, Parquet for processed (faster reads, typed schema)         |
| Data validation | **Pandera** for schema validation on the raw CSV at pipeline start      |

**Why Parquet for processed data?**  
Parquet preserves dtypes (especially ordinal categories), is 4–10x faster to read than CSV, and is the standard for ML pipelines. The raw CSV stays as CSV for transparency and easy inspection.

---

## 4. Machine Learning Stack

### 4.1 Feature Engineering

| Item                | Decision                                                            |
| ------------------- | ------------------------------------------------------------------- |
| Library             | **scikit-learn 1.4.x** Pipelines + ColumnTransformer                |
| Ordinal encoding    | `sklearn.preprocessing.OrdinalEncoder` with explicit category order |
| Log transforms      | `sklearn.preprocessing.FunctionTransformer(np.log1p)`               |
| Engineered features | `sklearn.preprocessing.FunctionTransformer` (custom)                |
| Serialization       | `joblib` — pipeline saved as `pipeline.joblib`                      |

**Rules:**
- The entire feature pipeline must be a single `sklearn.Pipeline` object
- The pipeline must be `fit` only on training data, never on the full dataset
- The fitted pipeline is saved to MLflow as an artifact alongside the model
- At inference time, the pipeline is loaded from MLflow — never from a local file path

**Ordinal category orders (must be exact):**

```python
CUT_ORDER    = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER  = ["J", "I", "H", "G", "F", "E", "D"]      # J=worst, D=best
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
```

### 4.2 Models

| Model                     | Library        | Version | Role                 |
| ------------------------- | -------------- | ------- | -------------------- |
| XGBoost Regressor         | `xgboost`      | 2.x     | Primary candidate    |
| LightGBM Regressor        | `lightgbm`     | 4.x     | Speed benchmark      |
| CatBoost Regressor        | `catboost`     | 1.x     | Categorical baseline |
| GradientBoostingRegressor | `scikit-learn` | 1.4.x   | Reference benchmark  |

**Why these four and not others?**

- **XGBoost:** Industry standard for tabular regression. Consistently wins Kaggle competitions on structured data. Tree-based so SHAP TreeExplainer works natively and is fast.
- **LightGBM:** Faster training than XGBoost on larger datasets. Good for running many Optuna trials quickly.
- **CatBoost:** Handles ordinal categoricals natively without encoding — useful as a cross-check against our manual OrdinalEncoder approach.
- **GBM Baseline:** scikit-learn's own GBM is slower but well-understood. Acts as a sanity check — if XGBoost doesn't beat this by a meaningful margin, something is wrong.

**Why not neural networks (MLP, TabNet)?**
- SHAP TreeExplainer only works efficiently with tree-based models. Neural network explainability via KernelExplainer is 100–1000x slower and not feasible for a real-time `/explain` endpoint.
- Diamond price prediction is a well-understood tabular task. Tree ensembles consistently match or outperform neural nets here.
- Keep it simple for v1.0. Neural models are a v2 consideration.

**Rules:**
- All models must implement scikit-learn's `fit` / `predict` interface (or be wrapped in an `sklearn.Pipeline`)
- No model is trained with the full dataset — always use `train_test_split(test_size=0.2, random_state=42)`
- `random_state=42` everywhere, no exceptions (reproducibility)
- GPU is not required or assumed — all training runs on CPU

### 4.3 Hyperparameter Tuning

| Item               | Decision                                                       |
| ------------------ | -------------------------------------------------------------- |
| Library            | **Optuna 3.x**                                                 |
| Trials             | 50 trials per model (time-boxed to 10 minutes if running slow) |
| Sampler            | `TPESampler` (Optuna default — Bayesian)                       |
| Pruner             | `MedianPruner` — kills bad trials early                        |
| MLflow integration | Each trial logged as a nested MLflow run via `MLflowCallback`  |
| Objective metric   | Minimize RMSE on validation set                                |

**Why Optuna over GridSearchCV or RandomSearch?**  
GridSearch is exhaustive and wasteful. RandomSearch is better but dumb — it doesn't learn from previous trials. Optuna's TPE sampler learns which hyperparameter regions are promising and focuses trials there. 50 Optuna trials typically outperform 500 random trials.

**Hyperparameter search space for XGBoost (primary model):**

```python
{
    "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
    "max_depth":        trial.suggest_int("max_depth", 3, 9),
    "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
    "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
}
```

### 4.4 Explainability

| Item              | Decision                                                       |
| ----------------- | -------------------------------------------------------------- |
| Library           | **SHAP 0.45.x**                                                |
| Explainer type    | `shap.TreeExplainer` (tree models only — fast, exact)          |
| Global output     | `shap.summary_plot` — beeswarm, saved as PNG                   |
| Local output      | `shap.waterfall_plot` — per prediction, returned as base64 PNG |
| Dependence output | `shap.dependence_plot` for `carat` — saved as PNG              |
| Artifact storage  | All plots logged to MLflow as run artifacts                    |

**Why TreeExplainer and not KernelExplainer?**  
KernelExplainer is model-agnostic but approximates SHAP values using sampling — it is 100–1000x slower than TreeExplainer and produces noisier values. Since every model in our stack is tree-based, TreeExplainer gives exact SHAP values in milliseconds. This is what makes real-time `/explain` feasible.

**Rules:**
- SHAP values are computed on a background dataset of 200 randomly sampled training rows (not the full training set — this controls memory and speed)
- The `TreeExplainer` object is instantiated once at API startup and cached — never re-created per request
- SHAP waterfall plots returned from the API are base64-encoded PNG, not file paths

### 4.5 Experiment Tracking

| Item            | Decision                                                            |
| --------------- | ------------------------------------------------------------------- |
| Library         | **MLflow 2.x**                                                      |
| Tracking server | Self-hosted via Docker (`ghcr.io/mlflow/mlflow`)                    |
| Backend store   | PostgreSQL (artifact metadata, run params, metrics)                 |
| Artifact store  | Local filesystem (`./mlruns/`) mapped as a Docker volume            |
| Model registry  | MLflow Model Registry — best model promoted to `Diamond/Production` |
| UI port         | `5000`                                                              |

**Why MLflow and not Weights & Biases?**  
W&B is excellent and cloud-hosted, but Diamond is a self-hosted, local-first project. MLflow runs entirely on-premises with no external dependency or API key required, matching the project's philosophy. MLflow's model registry also integrates directly with the API's model loading pattern (`mlflow.pyfunc.load_model`).

**Mandatory logging per run:**

```python
mlflow.log_params({...})           # All hyperparameters
mlflow.log_metrics({               # Evaluated on test set
    "rmse": ...,
    "mae":  ...,
    "r2":   ...,
    "mape": ...
})
mlflow.log_artifact("shap_summary.png")
mlflow.log_artifact("shap_dependence_carat.png")
mlflow.sklearn.log_model(model, "model")
mlflow.set_tag("model_type", "xgboost")
mlflow.set_tag("python_version", "3.11")
```

---

## 5. API Layer

### 5.1 Framework

| Item        | Decision                                                                |
| ----------- | ----------------------------------------------------------------------- |
| Framework   | **FastAPI 0.111.x**                                                     |
| ASGI server | **Uvicorn 0.29.x**                                                      |
| Validation  | **Pydantic v2** (not v1 — breaking changes, v2 is significantly faster) |
| API docs    | Auto-generated Swagger UI at `/docs`, ReDoc at `/redoc`                 |
| CORS        | Enabled for `localhost:8501` (Streamlit) only                           |
| Port        | `8000`                                                                  |

**Why FastAPI over Flask or Django REST?**  
FastAPI gives automatic OpenAPI docs, native Pydantic integration, async support, and type-checked request/response models out of the box. Flask requires manual validation wiring. Django REST is too heavy for a single-purpose prediction API.

**Why Pydantic v2 specifically?**  
Pydantic v2 is written in Rust and is 5–50x faster than v1 for validation. It also has stricter type enforcement by default — a `float` field will reject a string, not silently coerce it. For an ML API where bad inputs cause silent model errors, strict validation is essential.

### 5.2 API Structure

```
api/
├── main.py          # App factory, lifespan events (model loading)
├── schemas.py       # Pydantic v2 request + response models
├── routes/
│   ├── predict.py   # POST /predict
│   ├── explain.py   # POST /explain
│   └── health.py    # GET /health
├── services/
│   ├── model.py     # Model loading + inference logic
│   └── shap.py      # SHAP explainer wrapper
└── tests/
    ├── test_predict.py
    ├── test_explain.py
    └── test_health.py
```

**Rules:**
- The model and SHAP explainer are loaded **once** at startup via FastAPI's `lifespan` context manager — not on each request
- Route handlers contain zero business logic — they call service functions only
- All service functions are pure (no side effects other than logging)
- Request validation errors return HTTP `422` with field-level error detail (Pydantic default)
- Unhandled exceptions return HTTP `500` with a sanitised error message (no stack traces in responses)
- No `print()` statements — use Python's `logging` module with structured output

### 5.3 Pydantic v2 Input Schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class DiamondInput(BaseModel):
    model_config = {"strict": True}

    carat:   float = Field(..., ge=0.2,  le=5.01,  description="Weight in carats")
    cut:     Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color:   Literal["D", "E", "F", "G", "H", "I", "J"]
    clarity: Literal["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]
    depth:   float = Field(..., ge=43.0, le=79.0,  description="Depth percentage")
    table:   float = Field(..., ge=43.0, le=95.0,  description="Table percentage")
    x:       float = Field(..., ge=0.0,  le=31.8,  description="Length mm")
    y:       float = Field(..., ge=0.0,  le=58.9,  description="Width mm")
    z:       float = Field(..., ge=0.0,  le=31.8,  description="Depth mm")
```

---

## 6. Dashboard Layer

### 6.1 Framework

| Item        | Decision                                                          |
| ----------- | ----------------------------------------------------------------- |
| Framework   | **Streamlit 1.35.x**                                              |
| Charts      | **Plotly 5.x** (via `st.plotly_chart`) — not Matplotlib or Altair |
| HTTP client | **httpx 0.27.x** (async-capable, replaces `requests`)             |
| Port        | `8501`                                                            |

**Why Streamlit and not Dash, Gradio, or a React frontend?**

- **Streamlit:** Fastest path from Python model to interactive UI. No JavaScript required. For a solo one-week project this is the correct choice.
- **Dash:** More flexible but requires writing layout in Python callback syntax — significantly more boilerplate.
- **Gradio:** Optimised for demos, not multi-tab structured dashboards. Limited control over layout.
- **React:** Correct choice for v2 if the dashboard needs to be production-deployed. Overkill for v1.0.

**Why Plotly and not Matplotlib?**  
Plotly charts are interactive in Streamlit (hover, zoom, pan). Matplotlib charts are static PNGs. For SHAP waterfall and model comparison charts, interactivity is essential — hovering a bar to see exact values is core to the UX.

**Why httpx and not requests?**  
`httpx` supports both sync and async calls and has a nearly identical API to `requests`. Streamlit's execution model is synchronous, so both work, but `httpx` is the modern standard and will not need to be swapped if async is added later.

**Rules:**
- All API calls from the UI go through a single `api_client.py` module — never call `httpx` directly from a page file
- `st.session_state` is the only permitted state store — no global variables
- Default input values are defined in a single `constants.py` file — never hardcoded in UI files
- The UI must work even if the API is slow — use `st.spinner` for any operation over 300ms

---

## 7. Infrastructure & DevOps

### 7.1 Docker Compose

| Item            | Decision                                                                              |
| --------------- | ------------------------------------------------------------------------------------- |
| Compose version | Docker Compose V2 (`docker compose`, not `docker-compose`)                            |
| Network         | Single custom bridge network `diamond-net` — all services communicate by service name |
| Volumes         | Named volumes for PostgreSQL data and MLflow artifacts (not bind mounts)              |
| Health checks   | `postgres` and `mlflow` must be healthy before `api` starts                           |
| Restart policy  | `unless-stopped` for `postgres` and `mlflow`; `on-failure` for `api` and `ui`         |

**Service startup order:**

```
postgres  →  (healthy)  →  mlflow  →  (healthy)  →  api  →  ui
```

**Rules:**
- No service exposes ports to `0.0.0.0` except the ones meant for browser access (`5000`, `8000`, `8501`)
- PostgreSQL data is persisted in a named Docker volume — `docker compose down` does not destroy data; `docker compose down -v` does
- Each service has a `HEALTHCHECK` defined in its Dockerfile or Compose config

### 7.2 Makefile

All targets must be self-documenting:

```makefile
.PHONY: help up down train test ui mlflow clean

help:           ## Show this help
up:             ## Start all services (postgres, mlflow, api, ui)
down:           ## Stop all services
train:          ## Run training pipeline and log to MLflow
features:       ## Run feature engineering pipeline only
test:           ## Run pytest on API with coverage report
ui:             ## Open Streamlit dashboard in browser
mlflow:         ## Open MLflow UI in browser
clean:          ## Remove containers, volumes, and mlruns/
lint:           ## Run ruff + mypy
format:         ## Run ruff format
```

### 7.3 CI — GitHub Actions

**Trigger:** Push and PR to `main`

**Pipeline steps:**

```
1. Checkout
2. Set up Python 3.11
3. Install dependencies (pip install -e ".[dev]")
4. Run ruff (linter)
5. Run mypy (type checker)
6. Run pytest (API tests only — no Docker in CI)
   └── Coverage threshold: 80% minimum
7. Upload coverage report to job summary
```

**What CI does NOT do in v1.0:**
- Build Docker images (too slow for a personal project CI)
- Run training (requires GPU/long runtime)
- Deploy anywhere

**Rules:**
- CI must pass on every PR before merge — no exceptions
- The `main` branch is protected — direct pushes are not allowed
- Test failures block the PR — coverage below 80% blocks the PR

---

## 8. Code Quality

### 8.1 Linting & Formatting

| Tool           | Role                                                 | Config                    |
| -------------- | ---------------------------------------------------- | ------------------------- |
| **ruff**       | Linter + formatter (replaces flake8 + black + isort) | `pyproject.toml`          |
| **mypy**       | Static type checker                                  | `pyproject.toml`          |
| **pre-commit** | Runs ruff + mypy before every commit                 | `.pre-commit-config.yaml` |

**Why ruff over black + flake8?**  
ruff is written in Rust and runs 10–100x faster than the Python equivalents. It replaces black (formatting), flake8 (linting), and isort (import sorting) with a single tool and a single config block.

**ruff config in `pyproject.toml`:**

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore = ["ANN101", "ANN102"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

### 8.2 Testing

| Item         | Decision                                         |
| ------------ | ------------------------------------------------ |
| Framework    | **pytest 8.x**                                   |
| Coverage     | **pytest-cov** — minimum 80% on `api/`           |
| HTTP testing | **httpx** `TestClient` (FastAPI built-in)        |
| Fixtures     | Shared fixtures in `api/tests/conftest.py`       |
| Test data    | Fixed seed inputs defined in `tests/fixtures.py` |

**Rules:**
- Unit tests for all service functions
- Integration tests for all API endpoints (full request → response cycle)
- No tests that require a running Docker container (mock the model and SHAP explainer)
- Every test function name starts with `test_` and describes what it tests: `test_predict_returns_price_for_valid_input`

### 8.3 Logging

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
```

**Rules:**
- `INFO` level for normal operations (prediction received, model loaded)
- `WARNING` for recoverable issues (input near boundary, slow response)
- `ERROR` for failures that require attention (model load failed, DB unreachable)
- No `print()` anywhere in `src/` or `api/`
- Log prediction latency on every `/predict` call: `"Prediction completed in 42ms"`

---

## 9. Dependency Reference

Complete pinned dependency list by layer:

### Core ML
```
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.5
scikit-learn==1.4.2
optuna==3.6.1
mlflow==2.13.0
shap==0.45.0
numpy==1.26.4
pandas==2.2.2
pandera==0.19.3
joblib==1.4.2
pyarrow==16.0.0       # Parquet support
```

### API
```
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
httpx==0.27.0
python-multipart==0.0.9
python-dotenv==1.0.1
```

### Dashboard
```
streamlit==1.35.0
plotly==5.22.0
```

### Database
```
sqlalchemy==2.0.30
psycopg2-binary==2.9.9
```

### Dev / CI
```
pytest==8.2.0
pytest-cov==5.0.0
ruff==0.4.4
mypy==1.10.0
pre-commit==3.7.1
pip-tools==7.4.1
```

---

## 10. What We Are Explicitly Not Using

Documenting what was considered and rejected prevents re-litigating these decisions:

| Rejected Option | Considered For         | Reason Rejected                                |
| --------------- | ---------------------- | ---------------------------------------------- |
| Poetry          | Dependency management  | Heavier than pip-compile, custom lock format   |
| Conda           | Environment management | Heavy Docker layers, slow resolver             |
| Flask           | API framework          | Manual validation, no auto-docs                |
| Dash            | Dashboard              | More boilerplate than Streamlit for v1         |
| W&B             | Experiment tracking    | Requires external account; not self-hosted     |
| KernelExplainer | SHAP                   | 100–1000x slower than TreeExplainer            |
| SQLite          | MLflow backend         | Cannot handle concurrent writes                |
| TabNet / MLP    | Model selection        | SHAP TreeExplainer incompatible                |
| black + flake8  | Linting                | Replaced entirely by ruff                      |
| Redis           | Caching                | Overkill for v1.0 request volume               |
| Kubernetes      | Orchestration          | Docker Compose sufficient for local deployment |
| Alembic         | DB migrations          | MLflow manages its own schema                  |
| Celery          | Task queue             | No async training jobs in v1.0                 |
| React           | Dashboard frontend     | Overkill for v1.0 scope                        |

---

## 11. Version Upgrade Policy

- **Patch versions** (e.g. `2.0.3` → `2.0.4`): Upgrade freely, update `pyproject.toml`, re-run tests
- **Minor versions** (e.g. `2.0.x` → `2.1.x`): Review changelog for breaking changes, upgrade with a dedicated PR
- **Major versions** (e.g. `2.x` → `3.x`): Requires explicit decision and a migration plan — not done mid-sprint

---

## 12. Decision Log

| Date     | Decision                                  | Rationale                               | Decided By |
| -------- | ----------------------------------------- | --------------------------------------- | ---------- |
| Mar 2026 | Use PostgreSQL over SQLite for MLflow     | Concurrent write safety                 | Karthik    |
| Mar 2026 | Use TreeExplainer over KernelExplainer    | Real-time /explain endpoint feasibility | Karthik    |
| Mar 2026 | Use Streamlit over React for v1 dashboard | One-week scope constraint               | Karthik    |
| Mar 2026 | Use ruff over black + flake8              | Single tool, 10x faster, same output    | Karthik    |
| Mar 2026 | Use Pydantic v2 over v1                   | 5–50x faster validation, stricter types | Karthik    |
| Mar 2026 | Pin all deps to exact versions            | Reproducibility across machines and CI  | Karthik    |

---

*Tech Rules v1.0 — Diamond Project — March 2026*  
*All deviations require a documented justification in the PR description.*
