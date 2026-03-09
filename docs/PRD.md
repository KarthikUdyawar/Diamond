# 💎 Diamond — Product Requirements Document (PRD)

**Version:** 1.0  
**Author:** Karthik Udyawar  
**Repository:** [github.com/KarthikUdyawar/Diamond](https://github.com/KarthikUdyawar/Diamond)  
**Status:** In Planning  
**Target Completion:** 1 Week  

---

## 1. Overview

### 1.1 Background

The Diamond project currently exists as a Jupyter Notebook-based data analysis and price prediction system. It includes exploratory data analysis (EDA) and a baseline trained ML model, containerized via Docker to run notebooks. While functional as a research prototype, it lacks the infrastructure needed to be a production-grade, portfolio-worthy ML system.

### 1.2 Problem Statement

The current state of the project has three critical gaps:

- **No experiment tracking** — model runs are not logged, compared, or versioned
- **No serving layer** — predictions require running a notebook manually
- **No explainability** — there is no way to understand *why* a diamond received a specific price prediction

### 1.3 Goal

Transform Diamond from a notebook prototype into a complete, end-to-end ML product that demonstrates production-level thinking: better models, experiment tracking, explainability, a REST API, and an interactive dashboard — all wired together with Docker Compose.

---

## 2. Objectives

| #   | Objective                      | Success Metric                                               |
| --- | ------------------------------ | ------------------------------------------------------------ |
| O1  | Improve model accuracy         | R² > 0.98, RMSE < $550 on test set                           |
| O2  | Add experiment tracking        | All runs logged and comparable in MLflow UI                  |
| O3  | Add explainability             | SHAP plots generated per model run and per prediction        |
| O4  | Expose a prediction API        | `/predict` responds in < 100ms with price + confidence range |
| O5  | Build an interactive dashboard | All features accessible via a browser UI                     |
| O6  | One-command startup            | `make up` starts the entire stack                            |

---

## 3. Scope

### 3.1 In Scope

- Repo restructure into a production-grade folder layout
- Feature engineering pipeline (scikit-learn Pipeline)
- Model training with XGBoost, LightGBM, CatBoost, and Optuna tuning
- MLflow experiment tracking with PostgreSQL backend
- SHAP explainability (global summary + per-prediction waterfall)
- FastAPI prediction and explanation endpoints
- Streamlit dashboard with 3 tabs: Predictor, Explanation, Model Dashboard
- Docker Compose orchestration for all services
- GitHub Actions CI (pytest on API)
- Updated README with architecture diagram and screenshots

### 3.2 Out of Scope

- Real-time streaming data ingestion
- User authentication / multi-user support
- Cloud deployment (AWS / GCP / Azure)
- Mobile UI
- A/B model testing in production

---

## 4. Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────────────┐
│  Feature Engineering │  src/features.py
│  scikit-learn Pipeline│  Ordinal encoding, log transforms, volume features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐       ┌──────────────────────┐
│   Model Training     │──────▶│   MLflow Tracker      │
│   XGBoost / LightGBM│       │   PostgreSQL backend   │
│   CatBoost / Optuna  │       │   Model Registry       │
└─────────┬───────────┘       └──────────────────────┘
          │
          ▼
┌─────────────────────┐       ┌──────────────────────┐
│   FastAPI REST API   │──────▶│   SHAP Engine         │
│   /predict           │       │   TreeExplainer        │
│   /explain           │       │   Waterfall + Summary  │
│   /health            │       └──────────────────────┘
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Streamlit Dashboard │
│  Tab 1: Predictor    │
│  Tab 2: Explanation  │
│  Tab 3: Model Metrics│
└─────────────────────┘

All services orchestrated via Docker Compose
```

---

## 5. Folder Structure

```
Diamond/
├── data/
│   ├── raw/                    # Original CSV dataset
│   └── processed/              # Feature-engineered output
├── notebooks/
│   └── eda.ipynb               # Existing EDA (preserved, not modified)
├── src/
│   ├── __init__.py
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # Training loop + MLflow logging
│   ├── predict.py              # Inference logic (loads from MLflow registry)
│   └── explain.py              # SHAP logic
├── api/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # Pydantic v2 request/response models
│   └── tests/
│       └── test_api.py         # pytest API tests
├── ui/
│   └── app.py                  # Streamlit dashboard
├── mlruns/                     # MLflow artifacts (gitignored)
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 6. Feature Requirements

### 6.1 Feature Engineering Pipeline (`src/features.py`)

| Feature                   | Transformation           | Reason                       |
| ------------------------- | ------------------------ | ---------------------------- |
| `cut`, `color`, `clarity` | Ordinal encoding         | Natural ordered categories   |
| `carat`, `price`          | Log transform            | Right-skewed distributions   |
| `x`, `y`, `z`             | `volume = x * y * z`     | Captures 3D size             |
| `carat`, `volume`         | `carat_per_volume` ratio | Density proxy                |
| `table`, `depth`          | Keep as-is               | Already normally distributed |

The entire pipeline must be a single `sklearn.pipeline.Pipeline` object, serializable and reusable at inference time.

---

### 6.2 Model Training (`src/train.py`)

**Models to train:**

| Model              | Library    | Notes                         |
| ------------------ | ---------- | ----------------------------- |
| XGBoost Regressor  | `xgboost`  | Primary candidate             |
| LightGBM Regressor | `lightgbm` | Fast, strong baseline         |
| CatBoost Regressor | `catboost` | Handles categoricals natively |
| GBM Baseline       | `sklearn`  | Reference benchmark           |

**MLflow Logging Requirements — each run must log:**

- All hyperparameters via `mlflow.log_params()`
- Metrics: RMSE, MAE, R², MAPE via `mlflow.log_metrics()`
- Model artifact via `mlflow.sklearn.log_model()`
- SHAP summary plot as a PNG artifact
- Feature importance plot as a PNG artifact

**Optuna Integration:**

- Run 50 trials for the best-performing model
- Each trial logged as a nested MLflow run
- Best trial parameters automatically used for final model registration

**Model Registry:**

- Best model promoted to `models:/Diamond/Production` in MLflow registry
- API loads exclusively from the registry — no hardcoded model paths

---

### 6.3 SHAP Explainability (`src/explain.py`)

Three output types required:

**Global (per model run):**
- SHAP summary beeswarm plot — which features drive price across all diamonds
- Saved as `shap_summary.png`, logged to MLflow as artifact

**Local (per prediction):**
- SHAP waterfall plot — why this specific diamond got this price
- Returned as base64-encoded PNG from the `/explain` API endpoint

**Dependence (per model run):**
- SHAP dependence plot for `carat` vs `price`
- Saved as `shap_dependence_carat.png`, logged to MLflow as artifact

---

### 6.4 REST API (`api/main.py`)

**Base URL:** `http://localhost:8000`

#### `POST /predict`

**Request:**
```json
{
  "carat": 0.89,
  "cut": "Premium",
  "color": "H",
  "clarity": "SI2",
  "depth": 62.4,
  "table": 58.0,
  "x": 6.15,
  "y": 6.10,
  "z": 3.83
}
```

**Response:**
```json
{
  "predicted_price_usd": 3842.50,
  "confidence_range": {
    "low": 3540.00,
    "high": 4145.00
  },
  "model_version": "1",
  "mlflow_run_id": "abc123def456"
}
```

#### `POST /explain`

Same request body as `/predict`.

**Response:**
```json
{
  "shap_values": {
    "carat": 1243.20,
    "cut": 84.10,
    "color": -122.40,
    "clarity": 310.50,
    "depth": -18.20,
    "table": 9.80,
    "volume": 540.30,
    "carat_per_volume": 210.10
  },
  "base_value": 3812.00,
  "waterfall_plot_b64": "<base64 PNG string>"
}
```

#### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "model_name": "Diamond",
  "model_version": "1",
  "mlflow_run_id": "abc123def456"
}
```

**Validation Rules (Pydantic v2):**

| Field         | Type  | Constraints                                    |
| ------------- | ----- | ---------------------------------------------- |
| `carat`       | float | 0.2 – 5.01                                     |
| `cut`         | str   | One of: Fair, Good, Very Good, Premium, Ideal  |
| `color`       | str   | One of: D, E, F, G, H, I, J                    |
| `clarity`     | str   | One of: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF |
| `depth`       | float | 43.0 – 79.0                                    |
| `table`       | float | 43.0 – 95.0                                    |
| `x`, `y`, `z` | float | 0.0 – 31.8                                     |

---

### 6.5 Streamlit Dashboard (`ui/app.py`)

**Tab 1 — Price Predictor**
- Input controls: sliders for `carat`, `depth`, `table`, `x`, `y`, `z`; dropdowns for `cut`, `color`, `clarity`
- Live prediction on input change (calls `/predict`)
- Display: predicted price in large font, confidence range as a visual bar
- Plain-English summary: *"This diamond is estimated at $3,842 based on its 0.89ct weight and Premium cut."*

**Tab 2 — Explain This Prediction**
- Same input controls as Tab 1
- On "Explain" button click: calls `/explain`, renders SHAP waterfall chart
- Feature contribution table: sorted by absolute SHAP value
- Plain-English sentence per top 3 features: *"The high carat weight added $1,243 to the price."*

**Tab 3 — Model Dashboard**
- Pulls run data from MLflow tracking API
- Comparison table: all model runs with RMSE, MAE, R², training time
- Global SHAP summary plot from the best registered run
- Best model highlight with badge

---

## 7. Infrastructure

### 7.1 Docker Compose Services

| Service    | Image                   | Port | Purpose                   |
| ---------- | ----------------------- | ---- | ------------------------- |
| `postgres` | `postgres:15`           | 5432 | MLflow backend store      |
| `mlflow`   | `ghcr.io/mlflow/mlflow` | 5000 | Experiment tracking UI    |
| `api`      | Custom (Python 3.11)    | 8000 | FastAPI prediction server |
| `ui`       | Custom (Python 3.11)    | 8501 | Streamlit dashboard       |

### 7.2 Makefile Targets

| Command       | Action                               |
| ------------- | ------------------------------------ |
| `make up`     | Start all Docker Compose services    |
| `make down`   | Stop all services                    |
| `make train`  | Run training pipeline, log to MLflow |
| `make test`   | Run pytest on API                    |
| `make ui`     | Open Streamlit dashboard in browser  |
| `make mlflow` | Open MLflow UI in browser            |
| `make clean`  | Remove containers, volumes, mlruns   |

---

## 8. Non-Functional Requirements

| Requirement                    | Target        |
| ------------------------------ | ------------- |
| API response time (`/predict`) | < 100ms (p95) |
| API response time (`/explain`) | < 500ms (p95) |
| Model R² on test set           | > 0.98        |
| Model RMSE on test set         | < $550        |
| Docker Compose cold start time | < 60 seconds  |
| Test coverage on API           | > 80%         |

---

## 9. Development Roadmap

| Day | Milestone                                           | Deliverable                                        |
| --- | --------------------------------------------------- | -------------------------------------------------- |
| 1   | Repo restructure + MLflow + Docker Compose skeleton | `make mlflow` opens UI at :5000                    |
| 2   | Feature engineering pipeline                        | `src/features.py` with full sklearn Pipeline       |
| 3   | Model training + MLflow logging + Optuna            | 4+ runs visible in MLflow UI                       |
| 4   | SHAP explainability + artifact logging              | SHAP plots appear in MLflow run artifacts          |
| 5   | FastAPI REST API + Pydantic v2 validation           | `/predict` and `/explain` return correct responses |
| 6   | Streamlit dashboard (all 3 tabs)                    | Full UI working at :8501                           |
| 7   | CI pipeline + README + v1.0 tag                     | GitHub Actions green, README with screenshots      |

---

## 10. Release Criteria (v1.0)

All of the following must be true before tagging `v1.0`:

- [ ] R² > 0.98 on held-out test set
- [ ] MLflow UI shows at least 4 model runs with metrics and SHAP artifacts
- [ ] Best model registered in MLflow Model Registry under `Diamond/Production`
- [ ] `/predict` endpoint returns correct predictions with Pydantic validation
- [ ] `/explain` endpoint returns SHAP values and a base64 waterfall plot
- [ ] Streamlit dashboard renders all 3 tabs without errors
- [ ] `make up` starts the full stack from a clean clone
- [ ] GitHub Actions CI passes on `main` branch
- [ ] README includes architecture diagram, feature list, and setup instructions
- [ ] All secrets (DB credentials) handled via `.env` file, never hardcoded

---

## 11. Tech Stack Summary

| Layer                 | Technology                                |
| --------------------- | ----------------------------------------- |
| Language              | Python 3.11                               |
| Models                | XGBoost, LightGBM, CatBoost, scikit-learn |
| Hyperparameter Tuning | Optuna                                    |
| Experiment Tracking   | MLflow                                    |
| Explainability        | SHAP (TreeExplainer)                      |
| API Framework         | FastAPI                                   |
| Input Validation      | Pydantic v2                               |
| Dashboard             | Streamlit + Plotly                        |
| Database              | PostgreSQL 15                             |
| Containerization      | Docker + Docker Compose                   |
| CI                    | GitHub Actions                            |
| Package Management    | pyproject.toml + pip                      |

---

## 12. Open Questions

| #   | Question                                                                         | Owner   | Due   |
| --- | -------------------------------------------------------------------------------- | ------- | ----- |
| Q1  | Should the confidence range come from quantile regression or bootstrap sampling? | Karthik | Day 3 |
| Q2  | Should Optuna trials be capped at 50 or time-based (e.g., 10 min)?               | Karthik | Day 3 |
| Q3  | Should the Streamlit UI call the FastAPI directly or share inference code?       | Karthik | Day 5 |
| Q4  | Should `mlruns/` be committed to the repo or fully gitignored?                   | Karthik | Day 1 |

---

*PRD generated for Diamond v1.0 — March 2026*