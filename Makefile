# =============================================================================
# Diamond — Makefile
# Run `make help` to see all available targets.
# =============================================================================

.DEFAULT_GOAL := help

KAGGLE_DATASET := harshitlakhani/natural-diamonds-prices-images
RAW_DATA_DIR   := data/raw
COMPOSE        := docker compose

-include .env
export
# =============================================================================
# Help
# =============================================================================

.PHONY: help
help: ## Show all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' \
		| sort

# =============================================================================
# Environment setup
# =============================================================================

.PHONY: install
install: ## Install all dependencies with uv (creates .venv)
	uv sync --all-extras

.PHONY: install-dev
install-dev: ## Install deps + set up pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install
	@echo "✅ Dev environment ready."

.PHONY: env
env: ## Copy .env.example → .env (safe, won't overwrite existing)
	@if [ -f .env ]; then \
		echo "⚠️  .env already exists — not overwriting."; \
	else \
		cp .env.example .env; \
		echo "✅ .env created — fill in KAGGLE_USERNAME and KAGGLE_KEY."; \
	fi

# =============================================================================
# Data
# =============================================================================

.PHONY: download
download: ## Download raw dataset from Kaggle into data/raw/
	@echo "📥 Downloading dataset from Kaggle..."
	@if [ -z "$$KAGGLE_USERNAME" ] || [ -z "$$KAGGLE_KEY" ]; then \
		echo "❌ Set KAGGLE_USERNAME and KAGGLE_KEY in your .env first."; exit 1; \
	fi
	uv run kaggle datasets download \
		-d $(KAGGLE_DATASET) \
		-p $(RAW_DATA_DIR) \
		--unzip
	@echo "✅ Dataset downloaded to $(RAW_DATA_DIR)/"

# =============================================================================
# Docker — infrastructure
# =============================================================================

.PHONY: up
up: ## Start all services (postgres, mlflow, api, ui)
	$(COMPOSE) up -d
	@echo "✅ Stack is up:"
	@echo "   MLflow → http://localhost:5000"
	@echo "   API    → http://localhost:8000"
	@echo "   UI     → http://localhost:8501"

.PHONY: up-infra
up-infra: ## Start postgres + mlflow only (no api/ui — use during development)
	$(COMPOSE) up -d postgres mlflow
	@echo "✅ Infrastructure up — MLflow at http://localhost:5000"

.PHONY: down
down: ## Stop all services (data is preserved)
	$(COMPOSE) down

.PHONY: logs
logs: ## Tail logs for all services
	$(COMPOSE) logs -f

.PHONY: ps
ps: ## Show running service status
	$(COMPOSE) ps

# =============================================================================
# ML pipeline
# =============================================================================

.PHONY: features
features: ## Run feature engineering pipeline → data/processed/
	$(COMPOSE) run --rm api uv run python -m src.features

.PHONY: train
train: ## Run full training pipeline (features → train → log to MLflow)
	$(COMPOSE) run --rm api uv run python -m src.train

# =============================================================================
# Open in browser
# =============================================================================

.PHONY: mlflow
mlflow: ## Open MLflow UI in browser
	@xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || \
		echo "👉 Open http://localhost:5000 in your browser"

.PHONY: ui
ui: ## Open Streamlit dashboard in browser
	@xdg-open http://localhost:8501 2>/dev/null || open http://localhost:8501 2>/dev/null || \
		echo "👉 Open http://localhost:8501 in your browser"

.PHONY: api-docs
api-docs: ## Open FastAPI Swagger docs in browser
	@xdg-open http://localhost:8000/docs 2>/dev/null || open http://localhost:8000/docs 2>/dev/null || \
		echo "👉 Open http://localhost:8000/docs in your browser"

# =============================================================================
# Code quality
# =============================================================================

.PHONY: lint
lint: ## Run ruff (linter) + mypy (type checker)
	uv run ruff check src/ api/ ui/
	uv run mypy src/ api/ ui/

.PHONY: format
format: ## Auto-format and fix code with ruff
	uv run ruff format src/ api/ ui/
	uv run ruff check --fix src/ api/ ui/

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

# =============================================================================
# Tests
# =============================================================================

.PHONY: test
test: ## Run pytest with coverage report (≥80% required)
	uv run pytest api/tests/ src/tests/ \
		--cov=api --cov=src \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		-v

.PHONY: test-api
test-api: ## Run API tests only
	uv run pytest api/tests/ --cov=api --cov-report=term-missing -v

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Remove containers, volumes, mlruns/ ⚠️  destroys all data
	$(COMPOSE) down -v --remove-orphans
	rm -rf mlruns/
	@echo "✅ Clean complete."

.PHONY: clean-cache
clean-cache: ## Remove Python/tool caches only
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Caches cleared."

reset-mlflow: ## Wipe MLflow DB and artifacts — run make up-infra after
	docker compose down
	docker volume rm diamond_postgres-data diamond_mlflow-artifacts || true
