# AI Camera dashboard — CPU/general container (runs the full dashboard + CPU detection).
# Hailo acceleration needs HailoRT + /dev/hailo0 passthrough on the host and is not baked into this image.
FROM python:3.12-slim

# OpenCV + video runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
# Persist runtime data across restarts by mounting these (settings.json lives in /app):
VOLUME ["/app/events", "/app/ai-workspace"]
CMD ["python", "app.py"]
