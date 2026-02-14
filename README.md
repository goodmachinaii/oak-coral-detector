# OAK Coral Detector

Detección de objetos en tiempo real + profundidad (Z) con **Luxonis OAK** y aceleración en **Coral EdgeTPU**.

## Modelo real del sistema

- **Principal (Coral):** SSDLite MobileDet EdgeTPU
- **Fallback (CPU):** YOLOv4-tiny

El runtime muestra explícitamente el modo (`coral-docker`, `coral-local`, `cpu`) y el modelo activo en OSD.

## ¿Por qué SSDLite MobileDet y no YOLO en Coral?

| Modelo | Latencia Coral USB | mAP COCO | Ops en TPU |
|---|---:|---:|---:|
| SSDLite MobileDet | ~9 ms | 32.9% | 100% |
| YOLOv8n (320x320) | ~87-150 ms | ~37% | ~90% |
| YOLOv8s (320x320) | ~150+ ms | ~44% | ~90% |
| EfficientDet-Lite1 | ~56 ms | 34.3% | ~95% |

## Estructura

- `oak_coral_detector.py` (script principal)
- `start_coral_stack.sh`
- `stop_coral_stack.sh`
- `download_cpu_models.sh` (descarga fallback CPU)
- `docker-compose.yml`
- `docker/Dockerfile`
- `docker/app.py`
- `models/`

## Instalación rápida

```bash
git clone https://github.com/goodmachinaii/oak-coral-detector.git
cd oak-coral-detector
./download_cpu_models.sh   # opcional: fallback CPU
./start_coral_stack.sh
```

## Configuración por variables de entorno

### Host (`oak_coral_detector.py`)

- `OAK_CORAL_BASE_DIR`
- `OAK_CORAL_MODELS_DIR`
- `OAK_CORAL_VENV_ACTIVATE`
- `CORAL_DOCKER_URL`
- `CORAL_HTTP_TIMEOUT`
- `CONF_THRESHOLD`
- `NMS_THRESHOLD`
- `RGB_FPS`
- `RGB_PREVIEW_SIZE` (ej: `640,360`)

Ejemplo:

```bash
CONF_THRESHOLD=0.5 RGB_FPS=20 ./start_coral_stack.sh
```

### Backend Docker (`docker/app.py`)

- `CORAL_MODEL_PATH`
- `CORAL_LABELS_PATH`

## Verificación rápida

```bash
sudo docker compose ps
curl -s http://127.0.0.1:8765/health
```

## Notas

- La profundidad Z se estima con **mediana de ROI central** (más estable que 1 píxel).
- En OSD se muestra latencia de inferencia (`infer: X ms`) para monitorear degradación.
