FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/training/model.h5 \
    CLASS_NAMES_PATH=/app/artifacts/training/class_names.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py main.py dvc.yaml params.yaml ./
COPY config ./config
COPY templates ./templates
COPY artifacts ./artifacts

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
