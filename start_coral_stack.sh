#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Levanta backend Coral en Docker
sudo docker compose up -d --build

# Limpia flag y arranca app host
rm -f STOP_YOLO_PI_LUXONIS_CORAL.flag
source /home/machina/.openclaw/workspace/.venv-luxonis/bin/activate
exec python YOLO_PI_LUXONIS_CORAL.py
