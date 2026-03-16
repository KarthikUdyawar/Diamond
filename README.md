# 💎 Diamond

> End-to-end diamond price prediction — ML pipeline · REST API · Streamlit dashboard · MLflow experiment tracking

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/badge/package%20manager-uv-violet.svg)](https://github.com/astral-sh/uv)

---

## What Is This?

Diamond trains a **CatBoost regression model** on ~6,300 natural diamond records to predict price from physical characteristics (weight, shape, cut, colour, clarity, polish, symmetry, fluorescence, measurements). It exposes predictions via a **FastAPI REST API** with per-prediction **SHAP explanations**, tracked end-to-end in a **self-hosted MLflow** instance, and visualised in a **3-tab Streamlit dashboard**.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        docker-compose                        │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────┐    ┌────────┐  │
│  │ postgres │───▶│  mlflow  │───▶│  api   │───▶│   ui   │  │
│  │  :5432   │    │  :5000   │    │  :8000 │    │  :8501 │  │
│  └──────────┘    └──────────┘    └────────┘    └────────┘  │
│       │                │              │                     │
│  named volume     named volume   loads model           calls API
│  (postgres-data)  (mlflow-        from MLflow          via httpx
│                    artifacts)     registry                  │
└─────────────────────────────────────────────────────────────┘

ML Pipeline (runs via make train or uv run --env-file .env.local -m src.train):

  data/raw/              src/features.py       src/train.py
  Diamonds/              _merge_raw_csvs() ──▶  CatBoost + 3
  Diamonds2/             ColumnTransformer      baseline models
  (15 per-shape CSVs)    KNN impute        ──▶  Optuna 50 trials
                         Ordinal encode    ──▶  MLflow logging
                         OHE Shape         ──▶  SHAP artifacts
                         Engineer features ──▶  Model Registry
                                                Diamond/Production
```

---

## Prerequisites

| Tool              | Version | Install                                                           |
| ----------------- | ------- | ----------------------------------------------------------------- |
| Docker            | 24+     | [docs.docker.com](https://docs.docker.com/get-docker/)            |
| Docker Compose V2 | 2.x     | Included with Docker Desktop                                      |
| make              | any     | `sudo apt install make`                                           |
| uv                | latest  | `curl -LsSf https://astral.sh/uv/install.sh \| sh`                |
| Kaggle account    | —       | [kaggle.com](https://www.kaggle.com) — needed for `make download` |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/KarthikUdyawar/Diamond.git
cd Diamond

# 2. Set up environment files
cp .env.example .env.local
cp .env.example .env.staging
cp .env.example .env.testing
cp .env.local .env
# Edit .env.local — fill in KAGGLE_USERNAME and KAGGLE_KEY

# 3. Install dependencies
make install-dev

# 4. Download dataset
make download

# 5. Start infrastructure (postgres + mlflow)
make up-infra

# 6. Run feature engineering + training (local env)
make features
make train-local

# 7. Start full stack
make up

# 8. Open dashboard
make ui
```

---

## Environments

Three env files control behaviour across contexts. All are gitignored — only `.env.example` is committed.

| File           | `APP_ENV` | Log format        | MLflow URI       | Use               |
| -------------- | --------- | ----------------- | ---------------- | ----------------- |
| `.env.local`   | `local`   | Pretty (coloured) | `localhost:5000` | Local development |
| `.env.testing` | `testing` | Plain (no colour) | `localhost:5000` | pytest / CI       |
| `.env.staging` | `staging` | JSON              | `mlflow:5000`    | Docker Compose    |

`.env` is kept as an alias for `.env.local` for tools that expect a plain `.env`.

**Switching:**

```bash
uv run --env-file .env.local    -m src.features   # local
uv run --env-file .env.testing  -m pytest          # tests
# Docker Compose picks up .env.staging automatically via env_file:
```

---

## Make Targets

```text
make help          Show all targets
make install       Install all dependencies with uv
make install-dev   Install deps + pre-commit hooks
make env           Copy .env.example → .env (safe, won't overwrite)
make download      Download raw dataset from Kaggle

make up            Start all 4 services (staging env)
make up-infra      Start postgres + mlflow only
make down          Stop all services

make features      Run feature engineering pipeline (local env)
make train         Run training pipeline in Docker container (staging env)
make train-local   Run training pipeline locally (local env, needs make up-infra)

make test          Run pytest with coverage report (testing env, ≥ 80% required)
make test-api      Run API tests only (src/tests/api/)
make lint          Run ruff + mypy
make format        Auto-format with ruff

make mlflow        Open MLflow UI in browser
make ui            Open Streamlit dashboard in browser
make api-docs      Open FastAPI Swagger docs in browser

make clean         Remove containers, volumes, mlruns/ ⚠️ destroys all data
make clean-cache   Remove Python/tool caches
make reset-mlflow  Wipe MLflow DB and artifacts
```

---

## Project Structure

```text
Diamond/
├── src/                    # ML pipeline + API
│   ├── api/                # FastAPI REST API (Day 5)
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── routes/         # predict · explain · health
│   │   └── services/       # model · shap
│   ├── tests/
│   │   ├── api/            # API tests (Day 5)
│   │   ├── test_config.py
│   │   ├── test_logger.py
│   │   ├── test_explain.py
│   │   ├── test_features.py
│   │   └── test_train.py
│   ├── config.py           # Pydantic BaseSettings, get_settings() ✅
│   ├── logger.py           # structlog setup, get_logger() ✅
│   ├── explain.py          # SHAP ExplainerService ✅
│   ├── constants.py        # Column names, category orders, paths
│   ├── features.py         # Feature engineering pipeline ✅
│   └── train.py            # Model training + MLflow logging ✅
├── dockerfiles/
│   └── api.Dockerfile      # FastAPI service image
├── ui/                     # Streamlit dashboard (Day 6)
│   ├── app.py
│   ├── api_client.py
│   ├── constants.py
│   ├── templates.py
│   └── Dockerfile
├── data/
│   ├── raw/                # Kaggle download — Diamonds/ + Diamonds2/ subdirs
│   └── processed/          # train.parquet, test.parquet, pipeline.joblib (gitignored)
├── notebooks/              # EDA only — do not modify
├── docs/                   # PRD, Design Doc, Tech Rules, TODO, Day logs
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env                    # Alias for .env.local (gitignored)
├── .env.local              # Local development (gitignored)
├── .env.testing            # pytest / CI (gitignored)
├── .env.staging            # Docker Compose (gitignored)
└── .env.example            # Annotated template (committed)
```

---

## Dataset

**Source:** [Natural Diamonds Prices + Images](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images) — Kaggle

**Schema:**

| Column       | Type        | Description                                                    |
| ------------ | ----------- | -------------------------------------------------------------- |
| Shape        | categorical | Cushion, Emerald, Heart, Marquise, Oval, Pear, Princess, Round |
| Weight       | float       | Carats                                                         |
| Clarity      | ordinal     | I3 → FL (11 grades)                                            |
| Colour       | ordinal     | FANCY → D (20 grades)                                          |
| Cut          | ordinal     | Fair → Excellent (4 grades)                                    |
| Polish       | ordinal     | Fair → Excellent (4 grades)                                    |
| Symmetry     | ordinal     | Fair → Excellent (4 grades)                                    |
| Fluorescence | ordinal     | Very Strong → None (7 grades)                                  |
| Messurements | string      | `"L-W×D"` — parsed into length, width, depth_mm                |
| Price        | float       | Target — log1p-transformed during training                     |

---

## Model

**Primary model:** CatBoostRegressor, tuned with 50 Optuna trials (TPESampler + MedianPruner).

**Selected features (14 of 22):** Colour, Clarity, Polish, Symmetry, Fluorescence, Shape_Pear, Shape_Round, Weight, length, width, depth_mm, volume, carat_per_volume, log_weight.

**Registered:** `models:/Diamond/Production` in MLflow Model Registry.

| Run               | R² (log-space) |
| ----------------- | -------------- |
| catboost-baseline | 0.9674         |
| xgboost-baseline  | 0.9685         |
| lightgbm-baseline | 0.9656         |
| gbm-baseline      | 0.9661         |
| catboost-tuned    | best           |

**SHAP artifacts** (logged on tuned model only):

| Artifact                          | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `shap/shap_summary.png`           | Beeswarm — global feature impact distribution        |
| `shap/shap_dependence_weight.png` | SHAP vs Weight scatter with colour-coded interaction |
| `shap/shap_importance.png`        | Mean \|SHAP\| bar chart — ranked feature importance  |

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weight": 0.89,
    "shape": "Round",
    "cut": "Excellent",
    "colour": "H",
    "clarity": "SI2",
    "polish": "Excellent",
    "symmetry": "Very Good",
    "fluorescence": "None",
    "length": 6.15,
    "width": 6.10,
    "depth_mm": 3.83
  }'
```

**Response:**

```json
{
  "predicted_price_usd": 3842.0,
  "confidence_range": { "low": 3535.0, "high": 4149.0 },
  "model_version": "1",
  "mlflow_run_id": "abc123"
}
```

### `POST /explain`

Same request body as `/predict`. Returns SHAP values and a base64 waterfall plot.

### `GET /health`

```json
{ "status": "ok", "model_name": "Diamond", "model_version": "1", "mlflow_run_id": "abc123" }
```

---

## Development

```bash
make lint          # ruff + mypy
make format        # auto-format
make test          # pytest with coverage (testing env)
make pre-commit    # run pre-commit hooks on all files
```

**Note on running locally vs Docker:**

- `make train` runs inside the Docker api container with `.env.staging` (recommended for production runs)
- `make train-local` / `uv run --env-file .env.local -m src.train` runs locally — artifacts are uploaded via the MLflow HTTP proxy (`mlflow-artifacts:/` scheme routes through `MLFLOW_TRACKING_URI`)

---

## Git Flow

```text
main        ← protected, CI required, tagged releases only
  └── develop   ← integration branch
        └── feature/dayN-slug   ← daily work branches
```

Commit convention: `feat(day4): add SHAP explainability and structlog logging`

---

## Progress

| Day | Area                                                     | Status |
| --- | -------------------------------------------------------- | ------ |
| 1   | Repo restructure + Docker Compose + Makefile             | ✅ Done |
| 2   | Feature engineering pipeline                             | ✅ Done |
| 3   | Model training + MLflow + Optuna                         | ✅ Done |
| 4   | SHAP explainability + logging + config + api/ → src/api/ | ✅ Done |
| 5   | FastAPI REST API + tests                                 | ⬜ Next |
| 6   | Streamlit dashboard                                      | ⬜      |
| 7   | CI + README + v1.0 release                               | ⬜      |

---

## License

[MIT](LICENSE) — Karthik Udyawar · March 2026
