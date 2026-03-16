# dockerfiles/api.Dockerfile
# Diamond — FastAPI service
# Build context: project root (Diamond/)
#
# Reference in docker-compose.yml:
#   build:
#     context: .
#     dockerfile: dockerfiles/api.Dockerfile

FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.4.18

# Copy dependency manifest first (layer cache optimisation)
COPY pyproject.toml uv.lock ./

# Install all dependencies (frozen, no dev extras)
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ ./src/

# Env vars injected at runtime via env_file in docker-compose.yml
# Never bake secrets into the image.

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
