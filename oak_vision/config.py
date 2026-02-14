from __future__ import annotations
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    stop_file: Path
    log_file: Path
    models_dir: Path
    yolo_cfg: Path
    yolo_weights: Path
    yolo_names: Path
    coral_model: Path
    coral_labels: Path
    coral_docker_url: str
    coral_http_timeout: float
    coral_max_timeouts: int
    rgb_preview_size: tuple[int, int]
    rgb_fps: int
    conf_th: float
    nms_th: float
    headless: str
    db_path: Path
    db_retention_days: int
    db_prune_every_sec: int


def load_settings() -> Settings:
    base_dir = Path(os.environ.get('OAK_CORAL_BASE_DIR', Path(__file__).resolve().parent.parent))
    models_dir = Path(os.environ.get('OAK_CORAL_MODELS_DIR', base_dir / 'models'))
    return Settings(
        base_dir=base_dir,
        stop_file=base_dir / 'STOP_OAK_CORAL_DETECTOR.flag',
        log_file=base_dir / 'oak_coral_detector_runtime.log',
        models_dir=models_dir,
        yolo_cfg=models_dir / 'cpu/yolov4-tiny.cfg',
        yolo_weights=models_dir / 'cpu/yolov4-tiny.weights',
        yolo_names=models_dir / 'cpu/coco.names',
        coral_model=models_dir / 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
        coral_labels=models_dir / 'coco_labels.txt',
        coral_docker_url=os.environ.get('CORAL_DOCKER_URL', 'http://127.0.0.1:8765'),
        coral_http_timeout=float(os.environ.get('CORAL_HTTP_TIMEOUT', '2.0')),
        coral_max_timeouts=int(os.environ.get('CORAL_MAX_TIMEOUTS', '5')),
        rgb_preview_size=tuple(map(int, os.environ.get('RGB_PREVIEW_SIZE', '640,360').split(','))),
        rgb_fps=int(os.environ.get('RGB_FPS', '15')),
        conf_th=float(os.environ.get('CONF_THRESHOLD', '0.35')),
        nms_th=float(os.environ.get('NMS_THRESHOLD', '0.40')),
        headless=os.environ.get('HEADLESS', 'auto').lower(),
        db_path=Path(os.environ.get('OAK_DB_PATH', base_dir / 'data/oak.db')),
        db_retention_days=int(os.environ.get('OAK_DB_RETENTION_DAYS', '7')),
        db_prune_every_sec=int(os.environ.get('OAK_DB_PRUNE_EVERY_SEC', '300')),
    )


def log(settings: Settings, msg: str) -> None:
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{stamp}] {msg}'
    print(line, flush=True)
    settings.base_dir.mkdir(parents=True, exist_ok=True)
    with settings.log_file.open('a', encoding='utf-8') as f:
        f.write(line + '\n')
