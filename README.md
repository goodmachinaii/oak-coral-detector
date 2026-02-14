# YOLO PI LUXONIS CORAL (Con fallback a Pi)

Versión separada que prioriza Coral EdgeTPU con backend Docker, y fallback automático a CPU si Coral no está disponible.

## Arquitectura
- **Host (Pi):**
  - captura RGB + depth desde Luxonis
  - UI, botones STOP/EXIT, overlay, fusión con Z
- **Docker coral-infer:**
  - inferencia con `pycoral` + `tflite-runtime` compatibles (Debian Bullseye)
  - API local HTTP en `127.0.0.1:8765`

## Modos de inferencia (automático)
1. `coral-docker` (preferido)
2. `coral-local` (si existiera runtime local compatible)
3. `cpu` (fallback con YOLO host)

## Archivos principales
- `YOLO_PI_LUXONIS_CORAL.py`
- `docker-compose.yml`
- `docker/Dockerfile`
- `docker/app.py`
- `start_coral_stack.sh`
- `stop_coral_stack.sh`

## Modelos
- Host/Coral script:
  - `models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite`
  - `models/coco_labels.txt`
- Contenedor:
  - `docker/models/...` (mismos archivos)

## Controles UI
- **STOP / EXIT** (abajo izquierda)
- `ESC` / `q`
- Indicador en pantalla:
  - `INFERENCE: CORAL` (verde)
  - `INFERENCE: CPU FALLBACK` (rojo)

## Arranque recomendado
```bash
./start_coral_stack.sh
```

## Parada
```bash
./stop_coral_stack.sh
```

## Verificación de Coral real
```bash
sudo docker compose ps
curl -s http://127.0.0.1:8765/health
sudo docker exec coral-infer python3 - <<'PY'
from pycoral.utils.edgetpu import list_edge_tpus
print(list_edge_tpus())
PY
```
