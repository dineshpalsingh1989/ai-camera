# AI Camera dashboard — CPU/general container (runs the full dashboard + CPU detection).
# Hailo acceleration needs HailoRT + /dev/hailo0 passthrough on the host and is not baked into this image.
FROM python:3.12-slim

# OpenCV + video runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install CPU-only PyTorch first so pip does not pull multi-GB CUDA wheels (useless on Pi/CPU); then the rest.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# Configuration, statistics, inspection models and collected photos share a data volume.
# Keep the established event/workspace paths so existing named volumes still work.
ENV AI_CAMERA_DATA_DIR=/app/data
VOLUME ["/app/data", "/app/events", "/app/ai-workspace"]
CMD ["python", "app.py"]
