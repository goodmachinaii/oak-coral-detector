# Arquitectura Oak Vision Platform
## Propuesta modular: Inferencia · Almacenamiento · Front · OpenClaw
### Aditiva sobre github.com/goodmachinaii/oak-coral-detector

---

## 0. Supuestos (para ejecutar el plan)

- Hardware disponible y estable: Raspberry Pi 4 + Luxonis OAK + Coral USB.
- El flujo actual (`start_coral_stack.sh`) ya inicia detección funcional en `main`.
- El sistema seguirá corriendo en una sola máquina (sin despliegue distribuido).
- Se mantiene Docker para inferencia Coral (`docker/app.py`) y fallback CPU en host.
- El puerto de API local (`OAK_API_PORT`, default 5000) no colisiona con otros servicios.

---

## 1. ¿Un repo o varios?

**Un solo repo.** Todo corre en la misma Raspberry Pi 4. No hay
microservicios distribuidos, no hay escala horizontal. Un `git pull`
debe actualizar todo el sistema. Lo que sí se necesita es estructura
interna por módulos, no otro repositorio.

---

## 2. Estado actual del repo

```
oak-coral-detector/               ← 12 commits, rama main
├── oak_coral_detector.py          ← MONOLITO (~370 líneas)
├── docker-compose.yml             │  Captura OAK
├── start_coral_stack.sh           │  Inferencia Coral/CPU
├── stop_coral_stack.sh            │  Cálculo de profundidad
├── download_cpu_models.sh         │  Display + OSD
├── docker/                        │  Fallback + watchdog
│   ├── Dockerfile                 │  Todo junto en un archivo
│   └── app.py                     ← Flask backend Coral
├── models/
│   ├── ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite
│   ├── coco_labels.txt
│   └── cpu/
│       ├── yolov4-tiny.cfg
│       ├── yolov4-tiny.weights
│       └── coco.names
├── launchers/
│   └── START.desktop
├── .gitignore
└── README.md
```

El problema es `oak_coral_detector.py`. Hace 5 cosas distintas en un
solo archivo. No hay dónde insertar almacenamiento, API ni consultas
sin convertirlo en espagueti.

---

## 3. Estructura propuesta

Principio: **archivos nuevos, no reescritura**. El monolito se parte
en módulos pero la lógica probada se preserva. Los archivos existentes
(`docker/`, `models/`, scripts shell) no se tocan.

```
oak-coral-detector/
│
│   ── ARCHIVOS EXISTENTES (cambios mínimos compatibles) ──────────────────
│
├── docker/
│   ├── Dockerfile
│   └── app.py                     Flask backend Coral
├── models/
│   ├── coral/                     SSDLite MobileDet EdgeTPU
│   └── cpu/                       YOLOv4-tiny fallback
├── launchers/
│   └── START.desktop
├── docker-compose.yml
├── start_coral_stack.sh
├── stop_coral_stack.sh
├── download_cpu_models.sh
├── .gitignore
│
│   ── ARCHIVOS MODIFICADOS ───────────────────────────────
│
├── oak_coral_detector.py          SE CONVIERTE en orquestador delgado
│                                  (importa módulos, corre el loop)
├── README.md                      Se actualiza con nuevos módulos
│
│   ── ARCHIVOS NUEVOS ────────────────────────────────────
│
├── oak_vision/                    PAQUETE PYTHON
│   ├── __init__.py
│   ├── config.py                  Env vars centralizadas
│   ├── capture.py                 Pipeline OAK (DepthAI)
│   ├── inference.py               Coral HTTP + CPU DNN + fallback
│   ├── depth.py                   Profundidad mediana ROI
│   ├── storage.py                 SQLite + esquema + queries
│   ├── display.py                 Bounding boxes + OSD + headless
│   └── api.py                     REST API local (Flask ligero)
│
├── front/
│   └── index.html                 Dashboard SPA (un solo archivo)
│
├── openclaw/
│   ├── SKILL.md                   Skill para el agente LLM
│   └── queries.md                 Catálogo de queries SQLite
│
├── data/
│   └── .gitkeep                   SQLite se crea aquí en runtime
│
└── requirements.txt               Dependencias explícitas
```

---

## 4. Mapa de responsabilidades

Cada módulo tiene UNA responsabilidad. La comunicación es por
objetos Python en memoria, nunca por archivos temporales.

```
┌─────────────────────────────────────────────────────────┐
│                     MÓDULOS                              │
├──────────────┬──────────────────────────────────────────┤
│ config.py    │ Lee env vars, defaults, valida al inicio │
├──────────────┼──────────────────────────────────────────┤
│ capture.py   │ Pipeline DepthAI: entrega rgb + depth    │
├──────────────┼──────────────────────────────────────────┤
│ inference.py │ Envía frame a Coral o CPU, retorna       │
│              │ lista de (label, confidence, bbox)        │
├──────────────┼──────────────────────────────────────────┤
│ depth.py     │ Para cada bbox calcula depth_cm con      │
│              │ mediana ROI 30% central del depth map    │
├──────────────┼──────────────────────────────────────────┤
│ storage.py   │ INSERT en SQLite, limpieza periódica,    │
│              │ queries predefinidas para API/OpenClaw   │
├──────────────┼──────────────────────────────────────────┤
│ display.py   │ Dibuja boxes + depth + OSD en ventana    │
│              │ o se salta todo en modo headless         │
├──────────────┼──────────────────────────────────────────┤
│ api.py       │ REST en puerto local. Lee SQLite.        │
│              │ Sirve front/index.html como estático     │
└──────────────┴──────────────────────────────────────────┘
```

---

## 5. Flujo principal (loop por frame)

```
oak_coral_detector.py  ──  ORQUESTADOR
══════════════════════════════════════════════════════

  INICIO
    │
    ├─► config.load()            Lee env vars
    ├─► capture.init()           Abre pipeline OAK
    ├─► storage.init()           Crea/abre SQLite + schema
    ├─► api.start(storage)       Lanza Flask en thread aparte
    │
    ▼
  ╔══════════════╗
  ║  LOOP FRAME  ║◄─────────────────────────────┐
  ╚══════╤═══════╝                               │
         │                                       │
    ┌────▼─────────────┐                         │
    │   capture.grab() │                         │
    │   → rgb_frame    │                         │
    │   → depth_frame  │                         │
    └────┬─────────────┘                         │
         │                                       │
    ┌────▼──────────────────┐                    │
    │   inference.run(rgb)  │                    │
    │   → detections[ ]     │                    │
    │   → mode (coral/cpu)  │                    │
    │   → infer_ms          │                    │
    └────┬──────────────────┘                    │
         │                                       │
    ┌────▼──────────────────────┐                │
    │   depth.enrich(           │                │
    │       detections,         │                │
    │       depth_frame)        │                │
    │   → cada det ahora tiene  │                │
    │     depth_cm              │                │
    └────┬──────────────────────┘                │
         │                                       │
    ┌────▼──────────────────┐                    │
    │   storage.store(      │                    │
    │       detections,     │                    │
    │       mode, infer_ms) │                    │
    │   → INSERT en SQLite  │                    │
    └────┬──────────────────┘                    │
         │                                       │
    ┌────▼──────────────────┐                    │
    │   display.render(     │                    │
    │       rgb, detections,│                    │
    │       mode, infer_ms) │                    │
    │   → ventana/headless  │                    │
    └────┬──────────────────┘                    │
         │                                       │
         └───────────────────────────────────────┘
```

---

## 6. Diagrama de componentes y conexiones

```
┌───────────────────────────── RASPBERRY PI 4 ─────────────────────────────┐
│                                                                           │
│  ┌─────────────────── oak-coral-detector (repo) ────────────────────┐    │
│  │                                                                   │    │
│  │         oak_coral_detector.py (orquestador)                       │    │
│  │         ┌──────┬──────────┬────────┬──────────┬────────┐         │    │
│  │         │      │          │        │          │        │         │    │
│  │    ┌────▼──┐ ┌─▼────┐ ┌──▼───┐ ┌──▼────┐ ┌───▼──┐ ┌──▼──┐     │    │
│  │    │CAPTURE│ │INFER │ │DEPTH │ │STORAGE│ │DISPL │ │ API │     │    │
│  │    │.py    │ │.py   │ │.py   │ │.py    │ │.py   │ │.py  │     │    │
│  │    └───┬───┘ └──┬───┘ └──────┘ └──┬────┘ └──────┘ └──┬──┘     │    │
│  │        │        │                  │                   │         │    │
│  └────────┼────────┼──────────────────┼───────────────────┼─────────┘    │
│           │        │                  │                   │              │
│     ┌─────▼──┐  ┌──▼──────────┐  ┌───▼──────────┐  ┌────▼─────────┐   │
│     │LUXONIS │  │   DOCKER    │  │    SQLite     │  │  :5000/HTTP  │   │
│     │OAK CAM │  │ coral-infer │  │  data/oak.db  │  │  REST API    │   │
│     │RGB+Z   │  │ :8765/infer │  │              │  │              │   │
│     └────────┘  │ EdgeTPU     │  └──────────────┘  └──┬───┬───┬──┘   │
│                 └─────────────┘                        │   │   │      │
│                       │                                │   │   │      │
│                 ┌─────▼───┐                            │   │   │      │
│                 │CORAL USB│                            │   │   │      │
│                 │  TPU    │                            │   │   │      │
│                 └─────────┘                            │   │   │      │
└────────────────────────────────────────────────────────┼───┼───┼──────┘
                                                         │   │   │
                                      ┌──────────────────┘   │   │
                                      │    ┌─────────────────┘   │
                                      │    │    ┌────────────────┘
                                      ▼    ▼    ▼
                                 ┌─────┐ ┌───┐ ┌────────────┐
                                 │FRONT│ │LAN│ │  OPENCLAW   │
                                 │HTML │ │   │ │  (Telegram) │
                                 │:5000│ │   │ │  Lee API o  │
                                 └─────┘ │   │ │  sqlite3    │
                                         │   │ └────────────┘
                                         │   │
                                    ┌────▼───▼────┐
                                    │HOME ASSISTAN│
                                    │(futuro)     │
                                    └─────────────┘
```

---

## 7. Módulo STORAGE (detalle)

### 7.1 Schema SQLite

```sql
-- storage.py crea esto al inicializar

CREATE TABLE IF NOT EXISTS detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,   -- ISO 8601
    mode       TEXT    NOT NULL,   -- coral-docker | coral-local | cpu
    infer_ms   REAL,              -- latencia de inferencia
    label      TEXT    NOT NULL,   -- person, car, dog...
    confidence REAL    NOT NULL,   -- 0.0 a 1.0
    bbox_x     INTEGER,           -- esquina superior izquierda
    bbox_y     INTEGER,
    bbox_w     INTEGER,           -- ancho del bounding box
    bbox_h     INTEGER,           -- alto del bounding box
    depth_cm   REAL               -- DISTANCIA en centímetros
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_label     ON detections(label);
CREATE INDEX IF NOT EXISTS idx_depth     ON detections(depth_cm);

-- PRAGMAs para edge
PRAGMA journal_mode = WAL;       -- Write-Ahead Log
PRAGMA synchronous  = NORMAL;    -- balance seguridad/velocidad
PRAGMA cache_size   = -2000;     -- 2 MB cache en RAM
PRAGMA temp_store   = MEMORY;    -- temporales en RAM
```

### 7.2 Escritura (batch por frame)

```
Frame N detecta 3 objetos:

  storage.store(frame_result)
     │
     │  BEGIN TRANSACTION
     │    INSERT (person,  0.87, 120,45,180,290,  215.3)
     │    INSERT (car,     0.72, 400,200,250,180, 520.0)
     │    INSERT (dog,     0.65, 50,300,90,80,    185.7)
     │  COMMIT
     │
     │  Cada 5 minutos (o cada N frames):
     │    DELETE FROM detections
     │    WHERE timestamp < datetime('now', '-7 days')
```

Un INSERT por cada objeto detectado. La distancia (depth_cm) va
en cada fila porque cada objeto tiene su propia profundidad.
La transacción es una por frame completo (atómica).

### 7.3 Queries predefinidas

```python
# storage.py expone funciones, no SQL crudo

def get_latest()          # último frame de detecciones
def get_history(minutes)  # últimos N minutos
def get_stats(hours)      # conteo por label en período
def get_closest(label)    # objeto más cercano por tipo
def get_activity_hours()  # horas con más actividad
```

---

## 8. Módulo API (detalle)

Flask ligero corriendo en thread separado. Solo lectura de SQLite.
Cada thread usa su propia conexión SQLite (`check_same_thread=False` + timeout)
y consultas cortas para evitar bloqueos contra el writer.

```
GET /status
    → { "mode": "coral-docker", "uptime": 3600, "fps": 7.2 }

GET /detections
    → { "timestamp": "...", "objects": [...], "mode": "...", "infer_ms": 9.2 }

GET /detections/history?minutes=60
    → [ { "timestamp": "...", "objects": [...] }, ... ]

GET /detections/stats?hours=24
    → { "person": 142, "car": 38, "dog": 7,
        "closest_person_cm": 85.3, "busiest_hour": "14:00" }

GET /                      ← sirve front/index.html
GET /front/index.html      ← dashboard estático
```

No hay autenticación porque corre en LAN local.
Puerto configurable vía env var `OAK_API_PORT` (default 5000).

---

## 9. Módulo FRONT (detalle)

Un solo archivo HTML con JavaScript vanilla. Sin framework,
sin build, sin npm. Se sirve como estático desde api.py.

```
front/index.html
    │
    │  Consume:  GET /detections       (polling cada 2s)
    │            GET /detections/stats  (polling cada 30s)
    │
    │  Muestra:
    │
    │  ┌─────────────────────────────────────────────┐
    │  │         OAK CORAL DETECTOR - DASHBOARD       │
    │  ├──────────────────────┬──────────────────────┤
    │  │   DETECCIÓN ACTUAL   │   ESTADÍSTICAS 24H   │
    │  │                      │                      │
    │  │  person  87% 2.1m    │  person:  142        │
    │  │  car     72% 5.2m    │  car:      38        │
    │  │  dog     65% 1.8m    │  dog:       7        │
    │  │                      │                      │
    │  │  Modo: CORAL  9ms    │  Hora pico: 14:00   │
    │  ├──────────────────────┴──────────────────────┤
    │  │            HISTORIAL (última hora)           │
    │  │                                              │
    │  │  14:32  person(2) car(1)  más cercano: 1.5m │
    │  │  14:31  person(1)         más cercano: 3.2m │
    │  │  14:30  car(2) dog(1)     más cercano: 2.8m │
    │  │  ...                                         │
    │  └──────────────────────────────────────────────┘
    │
    │  Accesible desde cualquier dispositivo en la LAN:
    │     http://192.168.1.XX:5000/
```

---

## 10. Módulo OPENCLAW (detalle)

### 10.1 Dos modos de acceso

```
                    ┌──────────────┐
                    │   USUARIO    │
                    │  (Telegram/  │
                    │   WhatsApp)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   OPENCLAW   │
                    │   (agente)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │                         │
        MODO A: sqlite3           MODO B: API REST
        (acceso directo)          (acceso HTTP)
              │                         │
    ┌─────────▼──────────┐   ┌─────────▼──────────┐
    │ sqlite3 data/oak.db│   │ curl localhost:5000 │
    │ "SELECT ..."       │   │      /detections    │
    └─────────┬──────────┘   └─────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │  LLM recibe  │
                    │  texto plano │
                    │  interpreta  │
                    │  y responde  │
                    └──────────────┘
```

El Modo A (sqlite3 directo) es más eficiente en tokens porque
el output es tabular y compacto. El Modo B (API) es más simple
y no requiere que OpenClaw tenga acceso al filesystem.
El skill enseña ambos caminos.

### 10.2 Estructura del skill

```
openclaw/
├── SKILL.md             ← El agente lee esto
└── queries.md           ← Catálogo de queries

SKILL.md enseña:
  - Dónde está la DB:  data/oak.db
  - Qué contiene:      tabla detections con label, depth_cm, timestamp...
  - Cómo consultar:    sqlite3 o curl
  - Cuándo consultar:  solo cuando el usuario pregunta
  - Cómo interpretar:  depth_cm = distancia física al objeto
```

### 10.3 Ejemplo de interacción

```
Usuario (Telegram):  "qué ves?"

OpenClaw ejecuta:    sqlite3 data/oak.db \
                     "SELECT label, confidence, depth_cm \
                      FROM detections \
                      ORDER BY timestamp DESC LIMIT 10;"

SQLite responde:     person|0.87|215.3
                     person|0.82|340.1
                     car|0.72|520.0

OpenClaw al usuario: "Veo 2 personas, la más cercana a 2.1 metros,
                      y un carro a 5.2 metros."

Tokens consumidos:   ~300 (solo cuando se pregunta)
```

```
Usuario (Telegram):  "a qué hora hubo más movimiento hoy?"

OpenClaw ejecuta:    sqlite3 data/oak.db \
                     "SELECT strftime('%H:00',timestamp) as hora, \
                      COUNT(*) as n \
                      FROM detections \
                      WHERE date(timestamp)=date('now') \
                      GROUP BY hora ORDER BY n DESC LIMIT 3;"

SQLite responde:     14:00|47
                     09:00|35
                     18:00|28

OpenClaw al usuario: "Las horas con más actividad hoy fueron
                      las 2pm (47 detecciones), 9am (35) y 6pm (28)."
```

---

## 11. Cómo se parte el monolito (plan de migración)

El objetivo es NO romper nada que ya funciona.

```
FASE 1: Extraer módulos (no cambia comportamiento)
═══════════════════════════════════════════════════

  oak_coral_detector.py (370 líneas)
         │
         ├── líneas 1-43     ──►  config.py
         │   (env vars, constantes)
         │
         ├── líneas 44-230   ──►  capture.py + inference.py
         │   (OAK pipeline, coral HTTP, CPU DNN)
         │
         ├── líneas 269-320  ──►  depth.py
         │   (cálculo mediana ROI)
         │
         ├── líneas 321-360  ──►  display.py
         │   (OSD, bounding boxes, ventana)
         │
         └── lo que queda    ──►  oak_coral_detector.py
             (loop principal        ahora solo importa
              y control)            módulos y orquesta)


FASE 2: Agregar storage.py (nueva funcionalidad)
═══════════════════════════════════════════════════

  En el loop, después de depth.enrich():
    + storage.store(frame_result)

  El detector sigue funcionando igual,
  pero ahora también escribe en SQLite.


FASE 3: Agregar api.py + front/ (nueva funcionalidad)
═══════════════════════════════════════════════════════

  Al inicio del programa:
    + threading.Thread(target=api.start, daemon=True).start()

  Al cierre del programa:
    + señal de shutdown para detener API y cerrar conexiones limpias.

  El detector sigue funcionando igual,
  pero ahora hay un servidor web local accesible.


FASE 4: Agregar openclaw/ (solo documentación)
═══════════════════════════════════════════════════════

  Se copia SKILL.md y queries.md al workspace de OpenClaw.
  No modifica código del detector.
  Es solo un archivo de texto que enseña al agente.
```

---

## 12. Flujo de datos completo (end to end)

```
  HARDWARE                MÓDULOS PYTHON               CONSUMIDORES
 ═════════              ════════════════              ══════════════

                        ┌────────────┐
 ┌──────────┐  RGB      │            │
 │ LUXONIS  ├──────────►│  capture   │
 │ OAK      │  depth    │            │
 │ RGB+Z    ├──────────►│            │
 └──────────┘           └─────┬──────┘
                              │ (rgb, depth_frame)
                              │
                        ┌─────▼──────┐   HTTP    ┌─────────────┐
                        │            ├──────────►│ DOCKER      │
                        │ inference  │◄──────────┤ coral-infer │
                        │            │  JSON     │ (EdgeTPU)   │
                        └─────┬──────┘           └─────────────┘
                              │ (detections, mode, ms)
                              │
                        ┌─────▼──────┐
                        │   depth    │
                        │  enrich    │
                        └─────┬──────┘
                              │ (detections + depth_cm)
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
           ┌─────▼──────┐ ┌──▼──────┐ ┌────▼───────┐
           │  storage   │ │ display │ │            │
           │            │ │         │ │            │
           └─────┬──────┘ └─────────┘ │            │
                 │                     │            │
           ┌─────▼──────┐             │            │
           │  SQLite DB │             │            │
           │  data/     │             │            │
           │  oak.db    │             │            │
           └─────┬──────┘             │            │
                 │                     │            │
           ┌─────▼──────┐             │            │
           │   api.py   │             │            │
           │  :5000     │             │            │
           └──┬───┬───┬─┘             │            │
              │   │   │               │            │
              ▼   ▼   ▼               ▼            │
           ┌────┐│ ┌──────┐    ┌──────────┐        │
           │FRON││ │OPEN  │    │ VENTANA  │        │
           │HTML││ │CLAW  │    │ OPENCV   │        │
           │    ││ │      │    │ (local)  │        │
           └────┘│ └──────┘    └──────────┘        │
                 │                                  │
                 ▼                                  │
           ┌──────────┐                             │
           │  HOME    │                             │
           │ASSISTANT │                             │
           │(futuro)  │                             │
           └──────────┘                             │
```

---

## 13. Decisiones técnicas clave

### ¿Por qué Flask y no FastAPI para api.py?

El backend Coral (docker/app.py) ya usa Flask. Mantener
la misma dependencia en todo el proyecto simplifica.
FastAPI requiere uvicorn + async, y no hay beneficio
para un endpoint que sirve a 1-3 clientes en LAN.

### ¿Por qué no WebSocket para el front?

Polling cada 2 segundos es suficiente para un dashboard.
WebSocket agrega complejidad (threading, reconexión)
sin beneficio visible cuando el dato cambia 7 veces
por segundo pero el ojo humano no necesita más de 0.5 Hz
para leer texto en un panel.

### ¿Por qué SQLite y no JSON?

- Queries SQL nativos (COUNT, GROUP BY, MIN depth_cm)
- Integridad ante cortes de luz (WAL + transacciones)
- Sin límite artificial de rotación
- Frigate (referente del ecosistema Coral) usa SQLite
- OpenClaw puede ejecutar sqlite3 como comando shell
- sqlite3 viene preinstalado en Raspberry Pi OS

### ¿Dónde queda el archivo SQLite?

En `data/oak.db` dentro del repo. La carpeta `data/` va
en .gitignore. Es almacenamiento local persistente en
la SD card. El desgaste es negligible (~50 KB/hora de
escritura con retención de 7 días).

### ¿Y si se necesita RAM para la DB?

Env var `OAK_DB_PATH`. Default: `data/oak.db` (SD).
Si el usuario configura tmpfs o /dev/shm, la cambia:
  `OAK_DB_PATH=/dev/shm/oak.db`
Pero pierde datos al reiniciar. Para este volumen de
datos, la SD es perfectamente viable.

---

## 14. Resumen visual de lo que se AGREGA al repo

```
ANTES (lo que existe)          DESPUÉS (lo que se agrega)
═════════════════════          ═══════════════════════════

oak_coral_detector.py ───────► se refactoriza en módulos
docker/                ───────► sin cambios
models/                ───────► sin cambios
scripts shell          ───────► sin cambios
launchers/             ───────► sin cambios
                       ───────► + oak_vision/        (paquete)
                       ───────►   + config.py
                       ───────►   + capture.py
                       ───────►   + inference.py
                       ───────►   + depth.py
                       ───────►   + storage.py       ← NUEVO
                       ───────►   + display.py
                       ───────►   + api.py           ← NUEVO
                       ───────► + front/index.html   ← NUEVO
                       ───────► + openclaw/SKILL.md  ← NUEVO
                       ───────► + openclaw/queries.md
                       ───────► + data/.gitkeep      ← NUEVO
                       ───────► + requirements.txt   ← NUEVO
```

Cero repos nuevos. Cero dependencias pesadas nuevas.
Sin cambios estructurales en docker/ o models/ (solo ajustes menores de compatibilidad si hacen falta).
Principalmente código Python nuevo y un HTML estático.

---

## 15. Riesgos y mitigaciones

### Riesgo A: regresión funcional al partir el monolito
- **Mitigación:** migración por fases, tests manuales por fase, feature flag para volver al flujo actual.

### Riesgo B: bloqueos SQLite por concurrencia (writer + API reader)
- **Mitigación:** WAL activo, conexión por thread, timeout en conexión, consultas cortas y reintentos simples.

### Riesgo C: inestabilidad Coral/OAK (timeouts, device busy)
- **Mitigación:** fallback controlado a CPU, umbral de timeouts consecutivos, logs de inferencia y reinicio limpio.

### Riesgo D: degradación de rendimiento en Pi 4
- **Mitigación:** mantener polling front bajo (2s/30s), batch por frame en DB, limpieza periódica por ventana de tiempo.

### Riesgo E: crecimiento de DB en SD
- **Mitigación:** retención (7 días por default), tarea de limpieza periódica, opción `OAK_DB_PATH` para mover storage.

---

## 16. Criterios de aceptación (Definition of Done)

### Funcional
- El detector mantiene comportamiento actual (captura, inferencia, profundidad, OSD/headless).
- Si Coral falla, fallback CPU sigue funcionando sin colgar UI.
- API local responde: `/status`, `/detections`, `/detections/history`, `/detections/stats`.
- Front `front/index.html` muestra detección actual + estadísticas básicas.

### Datos
- SQLite `data/oak.db` se crea automáticamente.
- Inserciones por detección se registran con `timestamp`, `label`, `confidence`, `depth_cm`, `mode`, `infer_ms`.
- Retención limpia datos vencidos sin detener el loop principal.

### Operación
- `start_coral_stack.sh` y `stop_coral_stack.sh` siguen siendo el flujo oficial.
- README actualizado con arquitectura, env vars y troubleshooting.
- Todo implementado en rama nueva y mergeado a `main` solo tras validación end-to-end.

---
