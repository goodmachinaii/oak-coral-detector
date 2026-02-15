# OAK Coral Detector

Detección de objetos en tiempo real + profundidad (Z) con **Luxonis OAK** y aceleración en **Coral EdgeTPU**.

## Lenguajes del repositorio

![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=goodmachinaii&repo=oak-coral-detector&layout=compact&langs_count=8)

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

- `oak_coral_detector.py` (orquestador principal)
- `oak_vision/` (módulos fase 1/2/3: config, capture, inference, depth, display, storage, api)
- `front/index.html` (dashboard local)
- `data/oak.db` (SQLite runtime de detecciones)
- `openclaw/SKILL.md` + `openclaw/queries.md` (consultas para asistente)
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
- `CORAL_HTTP_TIMEOUT` (default 2.0s)
- `CORAL_MAX_TIMEOUTS` (default 5, timeouts consecutivos antes de fallback)
- `CONF_THRESHOLD`
- `NMS_THRESHOLD`
- `RGB_FPS`
- `RGB_PREVIEW_SIZE` (ej: `640,360`)
- `HEADLESS` (`auto` por defecto, o `1` para forzar sin GUI)
- `OAK_DB_PATH` (default: `data/oak.db`)
- `OAK_DB_RETENTION_DAYS` (default: `7`)
- `OAK_DB_PRUNE_EVERY_SEC` (default: `300`)
- `OAK_API_HOST` (default: `0.0.0.0`)
- `OAK_API_PORT` (default: `5000`)

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

## Arquitectura (diagramas)

### Flujo principal por frame

```text
oak_coral_detector.py (orquestador)

  config.load()
      |
  capture.grab() ---> rgb + depth
      |
  inference.run(rgb) ---> detections + mode + infer_ms
      |
  depth.enrich(detections, depth) ---> depth_cm por objeto
      |
  storage.store(...) ---> SQLite (data/oak.db)
      |
  display.render(...) ---> ventana (o headless)
      |
  api/status + dashboard (/)
```

### Componentes en la Raspberry Pi

```text
+---------------------------- Raspberry Pi -----------------------------+
|                                                                       |
|  oak_coral_detector.py                                                |
|      |                                                                |
|      +--> oak_vision.capture    --> Luxonis OAK (RGB + Depth)        |
|      +--> oak_vision.inference  --> coral-infer docker (:8765)       |
|      +--> oak_vision.depth      --> depth_cm (mediana ROI)           |
|      +--> oak_vision.storage    --> SQLite data/oak.db               |
|      +--> oak_vision.display    --> OpenCV GUI / headless            |
|      +--> oak_vision.api        --> HTTP local (:5000) + front       |
|                                                                       |
+-----------------------------------------------------------------------+
```

### Mapa de responsabilidades

```text
+------------+----------------------------------------------------------+
| Módulo     | Responsabilidad                                           |
+------------+----------------------------------------------------------+
| config.py  | Env vars, defaults y validación inicial                  |
| capture.py | Pipeline DepthAI (rgb + depth)                           |
| inference.py| Coral HTTP + fallback CPU, devuelve detecciones         |
| depth.py   | depth_cm por bbox (mediana ROI central)                  |
| storage.py | Persistencia SQLite + limpieza por retención             |
| display.py | OSD/bounding boxes + modo GUI/headless                   |
| api.py     | API local + front estático                               |
+------------+----------------------------------------------------------+
```

### Flujo end-to-end (hardware → módulos → consumidores)

```text
LUXONIS OAK (RGB+Depth)
          |
          v
      capture.py
          |
          v
     inference.py <------ coral-infer docker (:8765) <--- Coral USB TPU
          |
          v
       depth.py
          |
          +------> storage.py ------> data/oak.db ------> api.py (:5000)
          |                                                |        |
          |                                                |        +--> front/index.html
          |                                                |
          +------> display.py (OpenCV GUI/headless)       +--> OpenClaw (API o SQL)
```

### OpenClaw: modos de consulta

```text
Usuario -> OpenClaw
            |
            +--> Modo A: SQL directo (data/oak.db)
            |
            +--> Modo B: HTTP local (GET /status, /detections, /stats)
```

### Antes vs Después (resumen)

```text
ANTES: monolito único (oak_coral_detector.py)
DESPUÉS: orquestador + módulos oak_vision/ + SQLite + API + Front + OpenClaw docs
```

## Roadmap

### Iteración 1 (actual, estable) ✅

- [x] Renombre y limpieza de naming a OAK Coral Detector
- [x] Modularización fase 1 (`oak_vision/*`)
- [x] Storage SQLite fase 2 (`data/oak.db`)
- [x] API local + front fase 3
- [x] OpenClaw skill + queries fase 4
- [x] Hardening base: headless auto, fallback CPU, timeout streak

### Iteración 2 (siguiente)

- [ ] Paridad completa de endpoints entre backend Flask y fallback stdlib
- [ ] Tracking/deduplicación para métricas por evento (no solo por frame)
- [ ] Endpoints de historial y estadísticas avanzadas (closest, busiest hour)
- [ ] Mejor manejo de crash OAK/USB (backoff + autorecovery)
- [ ] Pruebas de regresión y checklist de release para merge a `main`

## Notas

- La profundidad Z se estima con **mediana de ROI central** (más estable que 1 píxel).
- En OSD se muestra latencia de inferencia (`infer: X ms`) para monitorear degradación.
