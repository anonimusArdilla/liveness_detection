FROM python:3.11-slim

WORKDIR /workspace/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
RUN mkdir -p /workspace/models /workspace/backend/data /workspace/backend/storage_data/accepted_photos

EXPOSE 8000

CMD ["sh", "-c", "python scripts/download_models.py --models-dir /workspace/models && alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
