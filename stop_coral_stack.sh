#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

touch STOP_OAK_CORAL_DETECTOR.flag
pkill -f "oak_coral_detector.py" || true
sudo docker compose down || true
