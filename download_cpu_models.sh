#!/usr/bin/env bash
set -euo pipefail

DIR="$(dirname "$0")/models/cpu"
mkdir -p "$DIR"

wget -nc -P "$DIR" https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
wget -nc -P "$DIR" https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget -nc -P "$DIR" https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

echo "Modelos CPU descargados en $DIR"
