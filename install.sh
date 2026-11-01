#!/bin/bash
# Install AI Camera as an auto-start service. Run from this folder as the normal user: bash install.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

sudo apt-get update
sudo apt-get install -y python3-opencv python3-flask python3-requests python3-numpy python3-yaml ffmpeg

python3 -c "import hailo_platform" 2>/dev/null \
  || echo "WARNING: HailoRT not found (Raspberry Pi: sudo apt install hailo-all). Video works, AI stays off."
[ -f /usr/local/hailo/resources/models/hailo8l/yolov8s.hef ] \
  || echo "WARNING: Hailo models not in /usr/local/hailo/resources/models/hailo8l/ - set paths in Settings > Model files."

sudo tee /etc/systemd/system/ai-camera.service >/dev/null <<EOF
[Unit]
Description=AI Camera dashboard
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=/usr/bin/python3 $DIR/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now ai-camera

python3 test_app.py
echo "Done. Open http://$(hostname -I | awk '{print $1}'):8080  (login admin / admin)"
