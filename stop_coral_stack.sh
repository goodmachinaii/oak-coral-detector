#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

touch STOP_YOLO_PI_LUXONIS_CORAL.flag
pkill -f "YOLO_PI_LUXONIS_CORAL.py" || true
sudo docker compose down || true
