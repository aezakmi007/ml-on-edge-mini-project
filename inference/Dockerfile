# LogiEdge Inference Service — Dockerfile
# Component D, Task D2
# Base image: python:3.11-slim (as required by assignment)

FROM python:3.11-slim

# ── Set working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies BEFORE copying model ──────────────────────────
# (All pip install layers must appear before COPY model.tflite — assignment rule)
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy inference service code ────────────────────────────────────────────────
COPY inference/inference_service.py .
COPY data_pipeline/training_stats.npy .

# ── Copy model file (this layer changes when model updates — OTA demo) ─────────
COPY inference/model.tflite .

# ── Environment variables (overridable at runtime) ─────────────────────────────
ENV MODEL_PATH=model.tflite
ENV TRUCK_ID=truck_001
ENV BROKER_HOST=host.docker.internal
ENV BROKER_PORT=1883

# ── Run the inference service ──────────────────────────────────────────────────
CMD ["python3", "inference_service.py"]