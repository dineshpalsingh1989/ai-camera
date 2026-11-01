# AI Camera

Self-hosted AI CCTV dashboard: live cameras, object detection and tracking, zones, line counting,
pose and fall detection, face blur, heatmap, motion, alerts with photos and clips, Telegram, analytics,
and quality inspection (anomaly detection, e.g. egg lines). Runs fully on the edge, no cloud.

- **Hailo-8 / 8L** (Raspberry Pi 5, CM5) runs detection, pose and faces on the AI chip.
- **Any PC without Hailo** falls back to CPU (Ultralytics YOLO), slower.

## Run on a PC (Windows / Linux)

```
python -m venv .venv
.venv/Scripts/pip install flask requests "opencv-python<5" numpy ultralytics   # Linux: .venv/bin/pip
.venv/Scripts/python app.py
```

Open http://127.0.0.1:8080 and log in with `admin` / `admin` (change it in Settings → Login).
Add cameras in Settings → Cameras (RTSP brands, USB webcam, HTTP stream, video file; **Find cameras** scans the network).

## Install on a Raspberry Pi with Hailo

```
bash install.sh
```

Installs dependencies, registers the `ai-camera` service (starts on boot) and runs the self-check.
Quality inspection additionally needs PyTorch (`pip install torch torchvision`).

## Demo videos

`python make_demo_eggs.py` creates synthetic egg-line videos in `demo-videos/` for trying inspection without a real line.

## Custom objects (e.g. egg)

Settings → AI detection → Custom objects. `yolov8s-worldv2.pt` finds objects by name with no training.
For production, collect photos (Settings → camera → Training photos), label them on Roboflow, export YOLOv8,
train, and set the `.pt` path as the model.

## Self-check

```
python test_app.py
```

`settings.json` (passwords), events, learned models and training photos stay local and are not committed.
