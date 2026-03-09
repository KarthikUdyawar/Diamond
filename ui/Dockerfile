FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (maximises Docker layer cache)
COPY pyproject.toml uv.lock* ./

# Install production deps only
RUN uv sync --frozen --no-dev

# Copy source
COPY ui/ ./ui/

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
