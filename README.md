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

```
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

ML Pipeline (runs inside api container via make train):

  data/raw/           src/features.py       src/train.py
  diamonds.csv   ──▶  ColumnTransformer ──▶  CatBoost + 3
  (Kaggle API)        KNN impute             baseline models
                      Ordinal encode    ──▶  Optuna 50 trials
                      OHE Shape         ──▶  MLflow logging
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

# 2. Set up environment
cp .env.example .env
# Edit .env — fill in KAGGLE_USERNAME and KAGGLE_KEY

# 3. Install dependencies
make install-dev

# 4. Download dataset
make download

# 5. Start infrastructure (postgres + mlflow)
make up-infra

# 6. Run feature engineering + training
make features
make train

# 7. Start full stack
make up

# 8. Open dashboard
make ui
```

---

## Make Targets

```
make help          Show all targets
make install       Install all dependencies with uv
make install-dev   Install deps + pre-commit hooks
make env           Copy .env.example → .env
make download      Download raw dataset from Kaggle
make up            Start all 4 services
make up-infra      Start postgres + mlflow only
make down          Stop all services
make features      Run feature engineering pipeline
make train         Run full training pipeline
make test          Run pytest with coverage report
make lint          Run ruff + mypy
make format        Auto-format with ruff
make mlflow        Open MLflow UI in browser
make ui            Open Streamlit dashboard in browser
make api-docs      Open FastAPI Swagger docs in browser
make clean         Remove containers, volumes, mlruns/
make clean-cache   Remove Python/tool caches
```

---

## Project Structure

```
Diamond/
├── src/                    # ML pipeline
│   ├── constants.py        # Column names, category orders, magic numbers
│   ├── features.py         # Feature engineering pipeline (Day 2)
│   ├── train.py            # Model training + MLflow logging (Day 3)
│   ├── explain.py          # SHAP explainability (Day 4)
│   └── tests/
├── api/                    # FastAPI REST API
│   ├── main.py             # App factory + lifespan
│   ├── schemas.py          # Pydantic v2 models
│   ├── routes/             # predict · explain · health
│   ├── services/           # model · shap
│   ├── Dockerfile
│   └── tests/
├── ui/                     # Streamlit dashboard
│   ├── app.py              # 3-tab app
│   ├── api_client.py       # All httpx calls
│   ├── constants.py        # Default inputs
│   ├── templates.py        # Plain-English sentence templates
│   └── Dockerfile
├── data/
│   ├── raw/                # diamonds.csv (from make download)
│   └── processed/          # train.parquet, test.parquet (gitignored)
├── notebooks/              # EDA only — do not modify
├── docs/                   # PRD, Design Doc, Tech Rules, TODO, Day logs
├── docker-compose.yml
├── pyproject.toml          # uv, all pinned deps, ruff + mypy config
├── Makefile
└── .env.example
```

---

## Dataset

**Source:** [Natural Diamonds Prices + Images](https://www.kaggle.com/datasets/harshitlakhani/natural-diamonds-prices-images) — Kaggle

**Schema:**

| Column       | Type        | Description                          |
| ------------ | ----------- | ------------------------------------ |
| Shape        | categorical | Round, Princess, Oval, … (8 shapes)  |
| Weight       | float       | Carats                               |
| Clarity      | ordinal     | I1 → IF (8 grades)                   |
| Colour       | ordinal     | J → D (7 grades)                     |
| Cut          | ordinal     | Fair → Ideal (5 grades)              |
| Polish       | ordinal     | Fair → Ideal (5 grades)              |
| Symmetry     | ordinal     | Fair → Ideal (5 grades)              |
| Fluorescence | ordinal     | Very Strong → None (5 grades)        |
| Messurements | string      | `"L x W x D"` — parsed into 3 floats |
| Price        | string      | Target — `"$3,842"` format           |

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weight": 0.89,
    "shape": "Round",
    "cut": "Premium",
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
# Lint + type check
make lint

# Auto-format
make format

# Run tests with coverage
make test

# Pre-commit (runs automatically on git commit)
make pre-commit
```

---

## Git Flow

```
main        ← protected, CI required, tagged releases only
  └── develop   ← integration branch
        └── feature/dayN-slug   ← daily work branches
```

Branch naming: `feature/day2-feature-pipeline`, `hotfix/mlflow-psycopg2`

Commit convention: `feat(day2): add KNN imputer for numerical columns`

---

## Progress

| Day | Area                                         | Status |
| --- | -------------------------------------------- | ------ |
| 1   | Repo restructure + Docker Compose + Makefile | ✅ Done |
| 2   | Feature engineering pipeline                 | ⬜ Next |
| 3   | Model training + MLflow + Optuna             | ⬜      |
| 4   | SHAP explainability                          | ⬜      |
| 5   | FastAPI REST API + tests                     | ⬜      |
| 6   | Streamlit dashboard                          | ⬜      |
| 7   | CI + README + v1.0 release                   | ⬜      |

---

## License

[MIT](LICENSE) — Karthik Udyawar · March 2026
