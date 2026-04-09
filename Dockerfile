FROM python:3.11-slim

# System deps for OpenCV and insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the InsightFace model at build time
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1, det_size=(640, 640)); print('Model ready')"

EXPOSE 8000

# Railway sets $PORT dynamically; fall back to 8000 locally
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
