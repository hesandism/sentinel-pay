# SentinelPay scoring API — Phase 4
# ==================================
# A small, single-purpose image that serves src/serve/api.py with uvicorn.
# The same image is reused (with a different command) to run the one-shot
# model registrar and to run the MLflow tracking server in docker-compose.

FROM python:3.11-slim

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
# - PYTHONUNBUFFERED: logs stream out immediately (good for `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# lightgbm needs libgomp (OpenMP) at runtime; curl is used by the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (layer caching): only re-runs when requirements change.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the code and the committed Phase-2 artifacts the app loads at startup.
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# api.py resolves PROJECT_ROOT from its own file location, so /app is correct.
# The container talks to sibling services (mlflow, redis) via compose DNS names,
# supplied through environment variables in docker-compose.yml.
EXPOSE 8000

# Default command: run the API. docker-compose overrides this for the registrar.
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
