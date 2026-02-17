# Oak Vision Platform — Iteración 2: Plan Completo

**Rama:** `feature/oak-vision-platform-modular`
**Hardware:** Raspberry Pi 4 + Luxonis OAK-D Lite + Coral USB TPU
**Fecha:** 15 febrero 2026

---

## El Problema Actual

Con 1 persona y 1 silla en la escena, el sistema reportó:

| Objeto | Detecciones DB | Realidad |
|--------|---------------|----------|
| person | 3,554 | 1 persona |
| chair | 903 | 1 silla |
| cell phone | 752 | NO existía |
| tv | 574 | NO existía |
| refrigerator | 432 | NO existía |
| snowboard/skis/cat... | 100+ | NO existían |

**Consecuencia directa para OpenClaw:** el agente lee `COUNT(*)` sobre detecciones crudas y reporta "3,554 personas detectadas". Es imposible dar respuestas útiles con datos así.

### Las 3 causas raíz

**Causa 1 — Sin tracking:** cada frame genera filas independientes. A ~7 FPS × 10 min = 4,200 filas por un solo objeto. No existe el concepto "es la misma persona que vi en el frame anterior".

**Causa 2 — Threshold bajo (0.35):** detecciones espurias como cell phone al 36%, snowboard al 38%, tv al 41% pasan el filtro. Cada una se multiplica por miles de frames.

**Causa 3 — Queries crudas:** `COUNT(*)` sobre `detections` no distingue "1 persona durante 10 minutos" de "3,554 personas distintas". No existe tabla `events`.

---

## Orden de Ejecución

| Fase | Qué | Corrige | Archivos |
|------|-----|---------|----------|
| 1 | Filtro de dos niveles (min_score + threshold) | Falsos positivos | config.py, event_tracker.py |
| 2 | Norfair tracker (Kalman) + tabla events | Inflación ×frame | event_tracker.py, storage.py, orchestrator, requirements |
| 3 | Centralizar queries en storage.py | SQL duplicado (ya sobre modelo nuevo) | storage.py, api.py |
| 4 | API/queries por evento + OpenClaw | Datos para agente | storage.py, api.py, queries.md |
| 5 | Hardening OAK + telemetría | Crashes y logs | orchestrator, storage.py |
| 6 | QA + merge a main | Regresión | tests/ |

**Nota sobre el orden:** Se invirtieron Fase 2 y 3 respecto al borrador original. Razón: si centralizamos queries (Fase 2 original) *antes* de tener el modelo events, tendríamos que refactorizar storage.py dos veces — primero con queries sobre detections y luego re-reescribirlas para events. Implementando primero el tracker + events (nueva Fase 2), las queries se centralizan una sola vez sobre el modelo final.

---

## Fase 1: Filtro de Dos Niveles (min_score + threshold) — Patrón Frigate

**Impacto:** elimina ~90% de falsos positivos sin tocar arquitectura.

### Fundamento: cómo lo hace Frigate (estándar de la industria)

Frigate NVR NO usa un solo threshold. Usa un sistema de dos niveles que es el estándar en producción para edge detection con modelos COCO:

- **`min_score`** (default Frigate: **0.5**) — piso absoluto. Cualquier detección individual por debajo se descarta inmediatamente como falso positivo. Nunca llega al tracker.
- **`threshold`** (default Frigate: **0.7**) — score computado sobre la mediana del historial de scores de un objeto trackeado. Solo cuando ese score cruza el threshold, el objeto se marca como "verdadero positivo" y genera un evento.

Para modelos Frigate+ (fine-tuned), los valores recomendados son aún más altos: person usa min_score 0.65 / threshold 0.85, car usa 0.65 / 0.85, dog usa 0.70 / 0.90.

Adicionalmente, Frigate combina esto con filtros geométricos (`min_area`, `max_area`, `min_ratio`, `max_ratio`) para descartar detecciones con formas imposibles (ej: una hoja detectada como perro).

### Por qué un solo threshold (nuestra propuesta anterior) no es suficiente

Con un threshold único de 0.50, una detección espuria al 0.52 que aparece en 3 frames crea un evento. Con el sistema de dos niveles, esa detección pasa el `min_score` y entra al tracker, pero su mediana nunca llega a 0.70, así que nunca se confirma como verdadero positivo. Es un filtro mucho más robusto contra falsos positivos intermitentes.

### Cambios en config.py

```python
# Antes
conf_th = float(os.environ.get('CONF_THRESHOLD', '0.35'))

# Después: sistema de dos niveles tipo Frigate
# Nivel 1: piso absoluto — detecciones debajo se descartan antes del tracker
min_score = float(os.environ.get('MIN_SCORE', '0.50'))

# Nivel 2: threshold de confirmación — mediana del historial de un track
#          debe superar este valor para que el objeto se marque como true positive
score_threshold = float(os.environ.get('SCORE_THRESHOLD', '0.70'))
```

### Integración con el EventTracker (Fase 2)

El sistema de dos niveles se implementa así:

```python
# En oak_coral_detector.py — ANTES del tracker:
# Nivel 1: descartar detecciones por debajo de min_score
filtered = [d for d in stored_rows if d['confidence'] >= settings.min_score]

# Pasar al tracker solo las que superan min_score
enriched = tracker.update(filtered)

# En event_tracker.py — TrackMetadata mantiene historial de scores:
@dataclass
class TrackMetadata:
    label: str
    score_history: list[float] = field(default_factory=list)
    confirmed: bool = False  # True cuando mediana >= threshold

    @property
    def computed_score(self) -> float:
        """Mediana del historial, padded a mínimo 3 valores (patrón Frigate)."""
        padded = self.score_history + [0.0] * max(0, 3 - len(self.score_history))
        return sorted(padded)[len(padded) // 2]

# En EventTracker.update():
#   Norfair asigna IDs y maneja Kalman prediction internamente.
#   TrackMetadata agrega el two-level threshold sobre Norfair:
#   meta.score_history.append(det['confidence'])
#   if not meta.confirmed and meta.computed_score >= self.score_threshold:
#       meta.confirmed = True  # ahora es true positive
#       → crear evento en DB
#   elif meta.confirmed:
#       → actualizar evento en DB
#   else:
#       → Norfair trackea pero NO genera evento en DB
```

### Flujo completo por frame

```
Detección cruda (ej: person 0.52, snowboard 0.38, person 0.87)
    │
    ▼ Nivel 1: min_score = 0.50
    │ snowboard 0.38 → DESCARTADO (nunca llega al tracker)
    │
    ▼ Norfair tracker (Kalman + IoU distance)
    │ person 0.52 → Track #1, score_history=[0.52]
    │ person 0.87 → Track #2, score_history=[0.87]
    │
    ▼ Nivel 2: threshold = 0.70 (mediana del historial)
    │ Track #1: mediana([0.52, 0.0, 0.0]) = 0.0 → NO confirmado, sin evento
    │ Track #2: mediana([0.87, 0.0, 0.0]) = 0.0 → NO confirmado aún
    │
    ▼ Después de 3+ frames con scores consistentes:
    │ Track #2: mediana([0.87, 0.85, 0.89]) = 0.87 → CONFIRMADO → crear evento
    │ Track #1: mediana([0.52, 0.48, 0.51]) = 0.51 → nunca confirma → sin evento
```

### Valores por defecto y comparación

| Parámetro | Nuestro default | Frigate default (COCO) | Frigate+ (fine-tuned) |
|-----------|----------------|----------------------|----------------------|
| min_score | 0.50 | 0.50 | 0.65 (person) |
| threshold | 0.70 | 0.70 | 0.85 (person) |

Nuestros defaults son idénticos a los de Frigate con modelo COCO, que es el mismo tipo de modelo que usamos (SSDLite MobileDet entrenado en COCO). Esto es deliberado: Frigate ha optimizado estos valores a lo largo de años de uso en producción con miles de usuarios.

### Filtros geométricos adicionales (opcional, recomendado)

Frigate también usa filtros por área y ratio del bounding box. Para Iteración 2 se puede agregar opcionalmente:

```python
# config.py — filtros geométricos opcionales
min_area = int(os.environ.get('MIN_DETECTION_AREA', '1000'))     # pixels²
max_ratio = float(os.environ.get('MAX_DETECTION_RATIO', '5.0'))  # width/height
```

Esto descartaría detecciones demasiado pequeñas (ruido) o con proporciones imposibles (un "person" más ancho que alto). No es crítico para Fase 1, pero es el siguiente paso natural.

### Resultado esperado

Con los mismos 10 minutos de escena (1 persona + 1 silla):
- Detecciones que pasan min_score (0.50): person y chair (genuinos) + algún espurio ocasional al ~0.52
- Eventos confirmados (threshold 0.70): **solo person y chair** (los espurios nunca acumulan scores suficientes)
- cell phone, tv, snowboard, etc.: **0 eventos** (descartados en Nivel 1 o nunca confirmados en Nivel 2)

La inflación por frame (múltiples filas por objeto) se resuelve adicionalmente en Fase 2 con la tabla events.

---

## Fase 2: Norfair Tracker (Kalman) + Tabla Events

**Este es el cambio central que convierte 3,554 filas en "1 evento person".**

### Nota: la OAK-D Lite tiene tracker en hardware (no aplica a nuestro caso)

La OAK-D Lite tiene un nodo `ObjectTracker` integrado (Kalman + Hungarian en el Myriad X) que asigna IDs persistentes. Sin embargo, este nodo requiere como input un nodo de detección OAK (`MobileNetDetectionNetwork`), no detecciones externas. Nuestro sistema usa el Coral USB como motor de inferencia vía HTTP, por lo que las detecciones llegan a Python, no al pipeline OAK. El ObjectTracker nativo no aplica a nuestra arquitectura.

### Decisión: Norfair (Kalman tracker) — por qué y no las alternativas

Se evaluaron 4 opciones de tracking en Python para nuestro escenario (cámara fija, 1-5 objetos, 7 FPS, Pi 4):

| Tracker | Técnica | Pros | Contras para nuestro caso |
|---------|---------|------|---------------------------|
| **IoU manual** | Overlap de bboxes entre frames | Cero dependencias, simple | Sin predicción de movimiento; fragmenta tracks ante gaps (benchmark: 10 tracks para 1 persona vs 3 con Norfair) |
| **SORT / ByteTrack** | Kalman + cascada de matching high/low score | Estado del arte en escenas densas (MOT17) | Optimizado para decenas/cientos de objetos con oclusiones frecuentes; overhead innecesario para 1-5 objetos; licencia GPL |
| **DeepSORT** | Kalman + embedding de apariencia (2ª red neuronal) | Mejor accuracy absoluta; re-ID tras oclusiones largas | Requiere modelo adicional de re-identificación → duplica carga computacional; inviable en Pi 4 con Coral ya ocupado; licencia GPL |
| **Norfair** ✅ | Kalman filter (basado en SORT) + distancia configurable | Kalman sin red extra; tolera gaps; `pip install norfair`; 15 líneas de integración; licencia BSD; diseñado para embedded | No tiene re-ID por apariencia (irrelevante con cámara fija y pocos objetos) |

**Decisión: Norfair.** Razones concretas:

1. **Kalman filter integrado** — predice posición futura del objeto, no solo compara IoU frame a frame. Mantiene el track aunque el detector falle 1-2 frames (parámetro `hit_counter_max`).
2. **`initialization_delay`** — implementa nativamente el patrón Frigate de confirmación: el objeto debe persistir N frames antes de ser considerado válido. Esto reemplaza código custom del two-level threshold para conteo de frames.
3. **Sin carga computacional extra** — a diferencia de DeepSORT, no requiere modelo de apariencia. El Kalman filter corre en CPU con overhead despreciable (~0.1ms por frame para 5 objetos).
4. **Diseñado para embedded** — dependencias mínimas (`pip install norfair`), funciona sin GPU, pre-instalación de OpenCV aprovechada si ya existe en el sistema.
5. **Licencia BSD** — permisiva, sin obligación de publicar código derivado (a diferencia de SORT/DeepSORT bajo GPL).
6. **Código reducido** — ~15 líneas de integración vs ~100 líneas de tracker IoU custom, menos surface de bugs.

Referencia: Norfair (Tryolabs) — https://github.com/tryolabs/norfair — Python 3.8+, mantenido activamente.

### Nuevo módulo: oak_vision/event_tracker.py

```python
"""Tracking con Norfair + two-level threshold → eventos confirmados."""
import time
import numpy as np
from dataclasses import dataclass, field

try:
    from norfair import Detection, Tracker
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False

# ─── Configuración por defecto ───────────────────────────────────────
DEFAULT_DISTANCE_THRESHOLD = 0.7   # IoU threshold para matching
DEFAULT_INITIALIZATION_DELAY = 3   # Frames para confirmar track (patrón Frigate)
DEFAULT_HIT_COUNTER_MAX = 15       # Frames sin detección antes de perder track
DEFAULT_SCORE_THRESHOLD = 0.70     # Nivel 2: mediana de scores para confirmar evento

# ─── Score history para two-level threshold ──────────────────────────
@dataclass
class TrackMetadata:
    """Metadata adicional por track: score history + estado de evento."""
    label: str
    score_history: list[float] = field(default_factory=list)
    confirmed: bool = False
    event_created: bool = False       # True una vez insertado en DB
    first_seen: float = 0.0
    depth_history: list[float] = field(default_factory=list)

    @property
    def computed_score(self) -> float:
        """Mediana del historial, padded a 3 valores mínimo (patrón Frigate)."""
        padded = self.score_history + [0.0] * max(0, 3 - len(self.score_history))
        return sorted(padded)[len(padded) // 2]

    @property
    def avg_depth_cm(self) -> float | None:
        valid = [d for d in self.depth_history if d is not None]
        return sum(valid) / len(valid) if valid else None

    @property
    def min_depth_cm(self) -> float | None:
        valid = [d for d in self.depth_history if d is not None]
        return min(valid) if valid else None

# ─── Función de distancia IoU para Norfair ───────────────────────────
def iou_distance(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Distancia basada en IoU para Norfair.
    Norfair espera distancia (menor = mejor match), así que retornamos 1 - IoU.
    detection.points  = [[x1,y1],[x2,y2]]
    tracked_object.estimate = [[x1,y1],[x2,y2]]
    """
    det_box = detection.points
    trk_box = tracked_object.estimate

    # Intersección
    ix1 = max(det_box[0][0], trk_box[0][0])
    iy1 = max(det_box[0][1], trk_box[0][1])
    ix2 = min(det_box[1][0], trk_box[1][0])
    iy2 = min(det_box[1][1], trk_box[1][1])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Unión
    area_det = (det_box[1][0] - det_box[0][0]) * (det_box[1][1] - det_box[0][1])
    area_trk = (trk_box[1][0] - trk_box[0][0]) * (trk_box[1][1] - trk_box[0][1])
    union = area_det + area_trk - inter

    iou = inter / union if union > 0 else 0.0
    return 1.0 - iou  # Norfair: menor distancia = mejor match


class EventTracker:
    """
    Wrapper sobre Norfair que agrega:
    - Two-level threshold (min_score + computed_score via mediana)
    - Metadata por track (score_history, depth, confirmed/event_created)
    - Interfaz de salida compatible con storage.py
    """

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        initialization_delay: int = DEFAULT_INITIALIZATION_DELAY,
        hit_counter_max: int = DEFAULT_HIT_COUNTER_MAX,
    ):
        if not NORFAIR_AVAILABLE:
            raise ImportError(
                "Norfair no instalado. Ejecutar: pip install norfair"
            )

        self.score_threshold = score_threshold
        self.tracker = Tracker(
            distance_function=iou_distance,
            distance_threshold=distance_threshold,
            initialization_delay=initialization_delay,
            hit_counter_max=hit_counter_max,
        )
        # Metadata indexada por norfair object.id
        self._meta: dict[int, TrackMetadata] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Recibe detecciones ya filtradas por min_score (Nivel 1).
        Retorna lista enriquecida con event_id, is_new_event, confirmed.

        Cada detección de entrada:
          {'bbox': (x,y,w,h), 'label': str, 'confidence': float,
           'depth_cm': float|None}

        Norfair asigna IDs persistentes. El two-level threshold
        (Nivel 2) decide si el track se confirma como evento real.
        """
        now = time.monotonic()

        # Convertir detecciones a formato Norfair: points = [[x1,y1],[x2,y2]]
        norfair_dets = []
        det_map = {}  # index → detección original

        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            points = np.array([[x, y], [x + w, y + h]], dtype=np.float32)
            norf_det = Detection(
                points=points,
                scores=np.array([det['confidence'], det['confidence']]),
                label=det['label'],
            )
            norfair_dets.append(norf_det)
            det_map[id(norf_det)] = det

        # Norfair tracking step
        tracked_objects = self.tracker.update(detections=norfair_dets)

        # Procesar resultados
        enriched = []
        for obj in tracked_objects:
            track_id = obj.id

            # Encontrar la detección original que matcheó con este object
            matched_det = None
            if obj.last_detection is not None:
                matched_det = det_map.get(id(obj.last_detection))

            if matched_det is None:
                # Norfair predijo posición sin detección nueva este frame
                continue

            # Inicializar o actualizar metadata
            if track_id not in self._meta:
                self._meta[track_id] = TrackMetadata(
                    label=matched_det['label'],
                    first_seen=now,
                )

            meta = self._meta[track_id]
            meta.score_history.append(matched_det['confidence'])
            if matched_det.get('depth_cm') is not None:
                meta.depth_history.append(matched_det['depth_cm'])

            # Nivel 2: ¿la mediana de scores supera threshold?
            was_confirmed = meta.confirmed
            if not meta.confirmed and meta.computed_score >= self.score_threshold:
                meta.confirmed = True

            # ¿Es nuevo evento? (confirmado por primera vez Y no creado en DB aún)
            is_new = meta.confirmed and not meta.event_created
            if is_new:
                meta.event_created = True

            # Extraer bbox del estimate de Norfair
            est = obj.estimate
            bbox = (int(est[0][0]), int(est[0][1]),
                    int(est[1][0] - est[0][0]), int(est[1][1] - est[0][1]))

            enriched.append({
                **matched_det,
                'bbox': bbox,
                'event_id': track_id,
                'is_new_event': is_new,
                'confirmed': meta.confirmed,
            })

        # Limpiar metadata de tracks que Norfair ya eliminó
        active_ids = {obj.id for obj in tracked_objects}
        dead_ids = [tid for tid in self._meta if tid not in active_ids]
        # No limpiar aquí — get_ended_tracks se encarga

        return enriched

    def get_ended_tracks(self) -> list[dict]:
        """
        Retorna tracks que Norfair ya eliminó (timeout) y que fueron
        confirmados como eventos. Limpia metadata.
        """
        active_ids = {obj.id for obj in self.tracker.tracked_objects}
        ended = []
        dead_ids = []

        for tid, meta in self._meta.items():
            if tid not in active_ids and meta.event_created:
                now = time.monotonic()
                ended.append({
                    'track_id': tid,
                    'label': meta.label,
                    'duration_sec': now - meta.first_seen,
                    'frame_count': len(meta.score_history),
                    'max_confidence': max(meta.score_history),
                    'avg_depth_cm': meta.avg_depth_cm,
                    'min_depth_cm': meta.min_depth_cm,
                })
                dead_ids.append(tid)
            elif tid not in active_ids:
                # Track murió sin confirmar → limpiar sin reportar
                dead_ids.append(tid)

        for tid in dead_ids:
            del self._meta[tid]

        return ended

    def get_active_summary(self) -> dict[str, int]:
        """Resumen: cuántos objetos confirmados activos por label."""
        active_ids = {obj.id for obj in self.tracker.tracked_objects}
        summary: dict[str, int] = {}
        for tid, meta in self._meta.items():
            if tid in active_ids and meta.confirmed:
                summary[meta.label] = summary.get(meta.label, 0) + 1
        return summary
```

**Parámetros clave y su función:**

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `distance_threshold=0.7` | IoU mínimo 30% (1-0.7) para match | Balance entre tolerancia a movimiento y evitar cross-matching |
| `initialization_delay=3` | 3 frames para inicializar track | Norfair no asigna ID hasta que el objeto persista 3 frames consecutivos. Filtra detecciones espurias de 1-2 frames |
| `hit_counter_max=15` | 15 frames sin detección = track muerto | A 7 FPS → ~2.1 segundos de tolerancia. Persona que se mueve brevemente fuera de FOV no pierde track |
| `score_threshold=0.70` | Mediana de scores ≥ 0.70 para confirmar | Two-level threshold Nivel 2: track existe en Norfair pero NO genera evento en DB hasta que mediana supere 0.70 |

**Relación entre `initialization_delay` y `score_threshold` (no son redundantes):**

Estos dos mecanismos filtran cosas distintas y operan en capas separadas:

- **`initialization_delay=3`** (capa Norfair): filtra por **persistencia espacial**. Un objeto debe aparecer en 3 frames consecutivos con IoU suficiente para que Norfair le asigne un ID. Elimina detecciones espurias de 1-2 frames que aparecen y desaparecen (ruido del detector). Opera *antes* de que nuestro código vea el tracked object.
- **`score_threshold=0.70`** (capa TrackMetadata): filtra por **calidad acumulada**. Un objeto puede persistir 20 frames pero con scores mediocres (0.50-0.55). La mediana nunca alcanza 0.70, así que nunca genera evento. Elimina objetos que el detector "ve" consistentemente pero con baja confianza (ej: una sombra detectada como "person" al 52% durante 30 frames).

**No hay doble latencia para objetos legítimos:** una persona real con score 0.85 pasa ambos filtros casi simultáneamente. En frame 3 Norfair asigna ID, y en frame 2-3 la mediana ya supera 0.70 (sorted([0.0, 0.85, 0.87])[1] = 0.85). El evento se crea en frame 3, con latencia de ~0.4 segundos a 7 FPS. Solo los objetos espurios experimentan "rechazo" por uno u otro filtro.

**Flujo de confirmación (dos niveles + Norfair):**

```
Detección raw (score 0.52)
  │
  ├─ Nivel 1: min_score=0.50 → PASA (no descartada)
  │
  ├─ Norfair: initialization_delay=3 → espera 3 frames
  │    Frame 1: score 0.52 → track inicializando...
  │    Frame 2: score 0.55 → track inicializando...
  │    Frame 3: score 0.51 → track ACTIVO (ID asignado)
  │
  ├─ Nivel 2: computed_score (mediana) = 0.52 < 0.70 → NO CONFIRMADO
  │    → Track existe en Norfair pero NO genera evento en DB
  │    → Si score nunca sube → track muere sin evento
  │
  └─ Si scores suben: [0.52, 0.55, 0.72, 0.75, 0.78]
       mediana = 0.72 ≥ 0.70 → CONFIRMADO → INSERT en events
```

### Nuevo schema SQLite: tabla events

**Nota de diseño:** `event_id` es autoincremental en DB (sobrevive reinicios del proceso). `tracker_track_id` almacena el ID que Norfair asignó internamente — es efímero y se reinicia con cada restart del tracker. La relación es: un `tracker_track_id` activo se mapea a un `event_id` persistente vía un dict en memoria (`_track_to_event`).

```sql
-- Agregar a SCHEMA_SQL en storage.py

CREATE TABLE IF NOT EXISTS events (
    event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tracker_track_id INTEGER,             -- ID de Norfair (efímero, se reinicia con el proceso)
    label            TEXT    NOT NULL,
    first_seen       TEXT    NOT NULL,
    last_seen        TEXT    NOT NULL,
    duration_sec     REAL,
    frame_count      INTEGER DEFAULT 1,
    max_confidence   REAL,
    min_depth_cm     REAL,
    avg_depth_cm     REAL,
    last_bbox_x      INTEGER,
    last_bbox_y      INTEGER,
    last_bbox_w      INTEGER,
    last_bbox_h      INTEGER,
    status           TEXT    DEFAULT 'active'  -- active | ended
);
CREATE INDEX IF NOT EXISTS idx_events_label  ON events(label);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_first  ON events(first_seen);

-- Agregar columna event_id a detections (nullable para backwards compat)
-- Se agrega via ALTER TABLE en init si no existe
ALTER TABLE detections ADD COLUMN event_id INTEGER REFERENCES events(event_id);
```

### Nuevas funciones en storage.py

```python
def create_event(self, tracker_track_id: int, label: str, bbox: tuple,
                 confidence: float, depth_cm: float | None) -> int:
    """Crear nuevo evento. Retorna event_id autoincremental de DB."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with self.conn:
        cur = self.conn.execute(
            "INSERT INTO events (tracker_track_id, label, first_seen, last_seen, "
            "frame_count, max_confidence, min_depth_cm, avg_depth_cm, "
            "last_bbox_x, last_bbox_y, last_bbox_w, last_bbox_h, status) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 'active')",
            (tracker_track_id, label, now_iso, now_iso, confidence,
             depth_cm, depth_cm, *bbox)
        )
        return cur.lastrowid  # event_id autoincremental

def update_event(self, event_id: int, bbox: tuple,
                 confidence: float, depth_cm: float | None) -> None:
    """Actualizar evento existente con nueva detección.
    NULL-safe: depth_cm=None no degrada min_depth_cm existente."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with self.conn:
        if depth_cm is not None:
            self.conn.execute(
                "UPDATE events SET last_seen=?, frame_count=frame_count+1, "
                "max_confidence=MAX(max_confidence, ?), "
                "min_depth_cm=MIN(min_depth_cm, ?), "
                "last_bbox_x=?, last_bbox_y=?, last_bbox_w=?, last_bbox_h=? "
                "WHERE event_id=?",
                (now_iso, confidence, depth_cm, *bbox, event_id)
            )
        else:
            # Sin depth: actualizar todo excepto min_depth_cm/avg_depth_cm
            self.conn.execute(
                "UPDATE events SET last_seen=?, frame_count=frame_count+1, "
                "max_confidence=MAX(max_confidence, ?), "
                "last_bbox_x=?, last_bbox_y=?, last_bbox_w=?, last_bbox_h=? "
                "WHERE event_id=?",
                (now_iso, confidence, *bbox, event_id)
            )

def close_event(self, event_id: int, duration_sec: float) -> None:
    """Marcar evento como terminado."""
    with self.conn:
        self.conn.execute(
            "UPDATE events SET status='ended', duration_sec=? WHERE event_id=?",
            (duration_sec, event_id)
        )
```

### Integración en oak_coral_detector.py

```python
# En main():
from oak_vision.event_tracker import EventTracker
tracker = EventTracker(
    score_threshold=settings.score_threshold,   # 0.70 default (Nivel 2)
    distance_threshold=0.7,                      # IoU ≥ 30% para match
    initialization_delay=3,                      # 3 frames para inicializar track
    hit_counter_max=15,                          # ~2.1s sin detección @ 7 FPS
)

# Mapping tracker_track_id → event_id (DB autoincremental)
# Necesario porque Norfair IDs son efímeros (se reinician con el proceso)
# mientras event_id en DB es permanente y autoincremental
track_to_event: dict[int, int] = {}

# En el loop de detección (run_once):
# Nivel 1: descartar por min_score (piso absoluto)
filtered = [d for d in stored_rows if d['confidence'] >= settings.min_score]

# Norfair asigna IDs + Nivel 2 evalúa confirmación
enriched = tracker.update(filtered)

for det in enriched:
    track_id = det['event_id']  # tracker_track_id de Norfair
    if det['is_new_event'] and det['confirmed']:
        # Track recién confirmado → crear evento en DB, obtener event_id real
        db_event_id = storage.create_event(
            track_id, det['label'], det['bbox'],
            det['confidence'], det.get('depth_cm'))
        track_to_event[track_id] = db_event_id
        det['event_id'] = db_event_id  # reemplazar con ID de DB
    elif det['confirmed'] and track_id in track_to_event:
        # Track ya confirmado → actualizar evento existente
        db_event_id = track_to_event[track_id]
        storage.update_event(db_event_id, det['bbox'],
                           det['confidence'], det.get('depth_cm'))
        det['event_id'] = db_event_id
    # else: track no confirmado → Norfair lo trackea pero NO genera evento

# Guardar detecciones raw con event_id (solo las confirmadas tienen evento)
storage.store(enriched, mode=mode, infer_ms=infer_ms)

# Cerrar eventos de tracks que Norfair expiró (hit_counter agotado)
for ended in tracker.get_ended_tracks():
    track_id = ended['track_id']
    if track_id in track_to_event:
        db_event_id = track_to_event.pop(track_id)
        storage.close_event(db_event_id, ended['duration_sec'])
```

### Resultado esperado con esta fase

Misma escena (1 persona + 1 silla, 10 minutos):

| | Antes (Iter. 1) | Después (Iter. 2) |
|---|---|---|
| **Tabla detections** | 3,554 filas person | ~4,200 filas person (con event_id) |
| **Tabla events** | (no existía) | **1 fila: person, 10min, active** |
| **OpenClaw pregunta "¿cuántas personas?"** | "3,554 personas" | **"1 persona (activa hace 10 min)"** |
| **Falsos positivos en events** | N/A | **0 eventos** (filtrados en Fase 1) |

---

## Fase 3: Centralizar Queries en storage.py

**Problema actual:** api.py tiene SQL inline en 2 lugares (Flask handlers + stdlib fallback). Cualquier cambio de schema requiere editar 4+ sitios. Ahora que las tablas events y detections ya existen (Fase 2), centralizamos las queries una sola vez sobre el modelo final.

### Nuevas funciones en DetectionStorage

```python
# storage.py — funciones de consulta centralizadas

def get_latest(self, limit=20) -> list[dict]:
    """Últimas detecciones crudas."""
    return self._query(
        "SELECT timestamp, mode, infer_ms, label, confidence, "
        "bbox_x, bbox_y, bbox_w, bbox_h, depth_cm "
        "FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    )

def get_history(self, minutes=60, label=None) -> list[dict]:
    """Historial temporal con filtro opcional por label."""
    sql = "SELECT * FROM detections WHERE timestamp >= datetime('now', ?)"
    params = [f'-{minutes} minutes']
    if label:
        sql += " AND label = ?"
        params.append(label)
    sql += " ORDER BY id DESC LIMIT 500"
    return self._query(sql, params)

def get_stats(self, hours=24) -> dict:
    """Estadísticas enriquecidas. Usa events si existe, sino detections."""
    # Si tabla events existe: cuenta eventos únicos
    # Si no: fallback a COUNT(*) sobre detections (backwards compatible)
    ...

def get_events(self, hours=24, label=None, status=None) -> list[dict]:
    """Eventos agregados (una fila = un objeto real en escena)."""
    ...

def get_event_detail(self, event_id: int) -> dict | None:
    """Un evento con todas sus detecciones asociadas."""
    ...

def get_closest(self, label=None) -> list[dict]:
    """Objeto más cercano por label (MIN depth_cm)."""
    ...

def _query(self, sql, params=()) -> list[dict]:
    """Helper interno, retorna list[dict]."""
    self.conn.row_factory = sqlite3.Row
    cur = self.conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]
```

### Cambio en api.py

Todos los endpoints (Flask y stdlib) llaman funciones de storage:

```python
# Antes (Flask):
rows = q("SELECT label, COUNT(*) as n FROM detections WHERE ...")

# Después (Flask y stdlib):
data = storage.get_stats(hours=hours)

# El constructor de ApiServer recibe storage en vez de db_path:
class ApiServer:
    def __init__(self, storage: DetectionStorage, state: dict, host='0.0.0.0', port=5000):
        self.storage = storage
        ...
```

El SQL queda SOLO en storage.py. Cero duplicación.

---

## Fase 4: API + Queries por Evento + OpenClaw

### Convención de timezone

La DB almacena timestamps en **UTC ISO 8601** (`datetime.now(timezone.utc).isoformat()`). Las queries internas (`datetime('now', '-24 hours')`) operan en UTC porque SQLite `datetime('now')` es UTC por defecto.

La API retorna timestamps en UTC. La conversión a hora local es responsabilidad del cliente (OpenClaw, dashboard, etc.). OpenClaw puede usar la timezone del sistema (`America/Bogota`, UTC-5) para mostrar horas legibles al usuario:

```python
# En OpenClaw o UI:
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+ (stdlib)

utc_ts = "2026-02-15T14:30:00+00:00"
local = datetime.fromisoformat(utc_ts).astimezone(ZoneInfo("America/Bogota"))
# → 2026-02-15 09:30:00-05:00
```

### Nuevos endpoints

```
GET /events?hours=24&label=person&status=active
→ [{"event_id": 1, "label": "person", "first_seen": "...",
    "duration_sec": 600, "frame_count": 4200, "max_confidence": 0.89,
    "min_depth_cm": 145.2, "status": "active"}]

GET /events/1
→ {"event_id": 1, "label": "person", ...,
   "detections": [{"timestamp": "...", "confidence": 0.87, ...}, ...]}

GET /stats?hours=24
→ {"hours": 24,
   "by_event": [{"label": "person", "event_count": 3, "total_duration_sec": 1800}],
   "by_detection": [{"label": "person", "n": 12600}],
   "active_now": {"person": 1, "chair": 1},
   "closest": [{"label": "person", "min_depth_cm": 145.2}],
   "busiest_hour": "14:00"}

GET /health
→ {"oak_connected": true, "coral_mode": "coral-docker",
   "uptime_sec": 3600, "last_error": null,
   "events_active": 2, "db_size_mb": 12.4}
```

### Paridad Flask / stdlib

Ambos backends (Flask y http.server stdlib) deben implementar semántica idéntica. Tabla canónica de endpoints:

| Endpoint | Método | Params | Descripción | Errores |
|----------|--------|--------|-------------|---------|
| `/status` | GET | — | Estado básico del sistema (legacy, backwards compat) | — |
| `/latest` | GET | `limit` (default 20) | Últimas N detecciones raw (debug) | — |
| `/events` | GET | `hours` (24), `label`, `status` | Eventos agregados | 400 si `hours` inválido |
| `/events/<id>` | GET | — | Detalle de 1 evento + sus detecciones | 404 si no existe |
| `/stats` | GET | `hours` (24) | Estadísticas: by_event, by_detection, active_now, closest | 400 si `hours` inválido |
| `/health` | GET | — | Salud del sistema: OAK, Coral, uptime, errores, DB size | — |

Reglas de paridad:

- Ambos backends llaman las mismas funciones de `storage.py` — la diferencia es solo el framework HTTP
- Misma estructura JSON en ambos (mismas keys, mismo orden)
- Errores retornan JSON: `{"error": "event not found"}` con código HTTP apropiado (400/404)
- `test_api.py` corre los mismos test cases contra ambos backends

### Actualización de openclaw/queries.md

```sql
-- NUEVO: Eventos únicos (lo que OpenClaw debe usar por defecto)
SELECT label, COUNT(*) AS event_count,
       SUM(duration_sec) AS total_seconds
FROM events
WHERE first_seen >= datetime('now', '-24 hours')
GROUP BY label ORDER BY event_count DESC;

-- NUEVO: ¿Qué hay activo ahora mismo?
SELECT label, event_id, first_seen, max_confidence, min_depth_cm
FROM events WHERE status = 'active';

-- NUEVO: ¿Cuánto tiempo estuvo la última persona?
SELECT label, first_seen, last_seen, duration_sec
FROM events WHERE label = 'person'
ORDER BY last_seen DESC LIMIT 1;

-- LEGACY: Conteo por frame (mantener para debug, NO usar para reportes)
SELECT label, COUNT(*) AS raw_frames
FROM detections
WHERE timestamp >= datetime('now', '-24 hours')
GROUP BY label ORDER BY raw_frames DESC;
```

### Instrucción clave para OpenClaw SKILL.md

Agregar al skill de OpenClaw:

```markdown
## Regla fundamental para reportar detecciones

SIEMPRE usar la tabla `events` para conteos y reportes al usuario.
NUNCA usar COUNT(*) sobre `detections` — eso cuenta frames, no objetos.

- "¿Cuántas personas hubo?" → SELECT COUNT(*) FROM events WHERE label='person'
- "¿Qué hay ahora?" → SELECT * FROM events WHERE status='active'
- "¿Cuánto tiempo estuvo?" → SELECT duration_sec FROM events WHERE...

La tabla `detections` solo se usa para debug o análisis granular por frame.
```

---

## Fase 5: Hardening OAK + Telemetría

### 5a. Reconexión OAK — patrón oficial Luxonis + backoff

El patrón documentado por Luxonis para manejar desconexiones es un loop exterior que **recrea `dai.Device()` completo** cada vez que hay error. No se intenta recuperar la conexión — se destruye y recrea:

```python
# Patrón oficial Luxonis (base):
pipeline = dai.Pipeline()
# ... configurar pipeline ...
while True:
    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue("name")
        while True:
            queue.get()
    # Si sale del with (error), el loop exterior recrea el Device
```

El código actual en `oak_coral_detector.py` ya sigue este patrón parcialmente (el `while True` en `main()` que llama `run_once()`), pero usa `time.sleep(1)` fijo entre reintentos. Lo mejoramos con backoff exponencial y telemetría:

```python
# oak_vision/hardening.py — nuevo módulo

class ExponentialBackoff:
    def __init__(self, initial=1.0, maximum=30.0, factor=2.0):
        self.initial = initial
        self.maximum = maximum
        self.factor = factor
        self._current = initial

    def wait(self):
        time.sleep(self._current)
        self._current = min(self._current * self.factor, self.maximum)

    def reset(self):
        self._current = self.initial
```

**Cambio en oak_coral_detector.py:**
```python
# Antes:
except Exception as e:
    log(settings, f'Reinicio de pipeline por excepción: {e}')
    time.sleep(1)  # ← fijo

# Después:
backoff = ExponentialBackoff(initial=1.0, maximum=30.0)

except Exception as e:
    log(settings, f'Reinicio de pipeline por excepción: {e}')
    storage.log_system_event('pipeline_restart', 'orchestrator', 'error', str(e))
    backoff.wait()  # 1s, 2s, 4s, 8s, 16s, 30s, 30s...

# Después de conexión exitosa (dentro de run_once, al obtener primer frame):
backoff.reset()
```

### 5b. Errores OAK típicos y watchdog

Los errores USB más comunes con OAK son `X_LINK_COMMUNICATION_NOT_OPEN` y `X_LINK_ERROR`, que típicamente indican cable USB malo o desconexión. El backoff exponencial evita saturar logs cuando estos ocurren en ráfaga.

Para ajustar la tolerancia del watchdog interno de la OAK (útil si hay latencia USB):
```bash
# Variable de entorno — default 4000ms para PoE, configurable
DEPTHAI_WATCHDOG=4500 python3 oak_coral_detector.py
```

Agregar a `config.py`:
```python
# Timeout del frame watchdog (no frames RGB → reinicio)
oak_frame_timeout = float(os.environ.get('OAK_FRAME_TIMEOUT', '6.0'))
```

### 5c. Tabla system_events para telemetría

```sql
CREATE TABLE IF NOT EXISTS system_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    event_type TEXT    NOT NULL,  -- pipeline_restart, coral_fallback,
                                 -- oak_timeout, coral_timeout, startup, shutdown
    component  TEXT,              -- oak, coral, orchestrator, api
    severity   TEXT,              -- info, warning, error
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_sysev_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sysev_ts   ON system_events(timestamp);
```

```python
# storage.py
def log_system_event(self, event_type: str, component: str,
                     severity: str, detail: str = '') -> None:
    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with self.conn:
        self.conn.execute(
            "INSERT INTO system_events (timestamp, event_type, component, severity, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (now_iso, event_type, component, severity, detail)
        )
```

### Eventos a registrar

| Momento | event_type | severity |
|---------|-----------|----------|
| Inicio del sistema | startup | info |
| Parada limpia | shutdown | info |
| Reinicio de pipeline | pipeline_restart | error |
| Coral timeout (individual) | coral_timeout | warning |
| Fallback Coral → CPU | coral_fallback | warning |
| No frames RGB >6s | oak_timeout | error |
| Reconexión exitosa | oak_reconnect | info |

### 5c.1 Gestión de carga de escritura (I/O en SQLite)

Con 3 tablas recibiendo writes (detections ~7/s, events ~ocasional, system_events ~raro), el I/O en micro-SD del Pi 4 puede ser un cuello de botella. Medidas:

**Batch de detections:** En lugar de 1 INSERT por frame, acumular N detecciones en memoria y hacer 1 INSERT con `executemany()` cada ~1 segundo. Parámetro configurable:

```python
# config.py
detection_flush_interval = float(os.environ.get('DETECTION_FLUSH_SEC', '1.0'))
detection_flush_batch = int(os.environ.get('DETECTION_FLUSH_BATCH', '10'))
```

**WAL mode:** Activar `PRAGMA journal_mode=WAL` en init de storage.py. Permite reads concurrentes con writes sin bloqueo (la API lee mientras el orchestrator escribe).

**Retención de system_events:** Limpieza automática de registros antiguos para que la tabla no crezca indefinidamente:

```python
def prune_system_events(self, max_age_days: int = 30) -> int:
    """Eliminar system_events más antiguos que max_age_days. Retorna filas eliminadas."""
    with self.conn:
        cur = self.conn.execute(
            "DELETE FROM system_events WHERE timestamp < datetime('now', ?)",
            (f'-{max_age_days} days',)
        )
        return cur.rowcount
```

Llamar en startup o como tarea periódica (cada 24h).

### 5d. Endpoint /health

```python
GET /health →
{
    "oak_connected": true,
    "coral_mode": "coral-docker",
    "uptime_sec": 3600,
    "fps": 7.2,
    "events_active": 2,
    "last_errors": [
        {"timestamp": "...", "event_type": "coral_timeout", "detail": "..."}
    ],
    "db_size_mb": 12.4
}
```

**`db_size_mb` incluye** el archivo principal `.db` más los archivos WAL (`.db-wal` y `.db-shm`) si existen, ya que con `PRAGMA journal_mode=WAL` el WAL puede crecer significativamente antes de un checkpoint:

```python
def get_db_size_mb(self) -> float:
    """Tamaño total de DB incluyendo WAL/SHM."""
    import os
    db_path = self.db_path
    total = os.path.getsize(db_path)
    for suffix in ('-wal', '-shm'):
        wal_path = db_path + suffix
        if os.path.exists(wal_path):
            total += os.path.getsize(wal_path)
    return round(total / (1024 * 1024), 1)
```

OpenClaw puede usar esto para responder "¿ha habido problemas con la cámara?".

---

## Fase 6: QA + Merge

### Tests unitarios (sin hardware)

```
tests/
├── test_event_tracker.py    # Norfair tracker, two-level threshold, lifecycle
├── test_storage.py          # Todas las funciones, eventos, telemetría
├── test_api.py              # Todos los endpoints con mock DB
└── test_config.py           # Defaults, env vars
```

**test_event_tracker.py — casos clave:**
```python
def test_same_object_same_event():
    """Misma bbox con IoU alto = mismo event_id."""
    tracker = EventTracker(score_threshold=0.7, initialization_delay=0)
    det = {'label': 'person', 'bbox': (100,100,50,100), 'confidence': 0.8, 'depth_cm': None}
    r1 = tracker.update([det])
    det2 = {'label': 'person', 'bbox': (102,101,50,100), 'confidence': 0.85, 'depth_cm': None}
    r2 = tracker.update([det2])
    assert r1[0]['event_id'] == r2[0]['event_id']

def test_different_objects_different_events():
    """Bboxes sin overlap = event_ids distintos."""
    tracker = EventTracker(score_threshold=0.7, initialization_delay=0)
    r = tracker.update([
        {'label': 'person', 'bbox': (0,0,50,100), 'confidence': 0.8, 'depth_cm': None},
        {'label': 'person', 'bbox': (400,400,50,100), 'confidence': 0.7, 'depth_cm': None},
    ])
    assert r[0]['event_id'] != r[1]['event_id']

def test_hit_counter_max_closes_track():
    """Track sin actualizar después de hit_counter_max frames = ended."""
    tracker = EventTracker(
        score_threshold=0.5,  # bajo para confirmar rápido
        initialization_delay=0,
        hit_counter_max=2,  # muere tras 2 frames sin detección
    )
    # Frame 1: crear track
    tracker.update([{'label': 'person', 'bbox': (100,100,50,100),
                     'confidence': 0.8, 'depth_cm': None}])
    # Frames 2-4: vacíos → Norfair decrementa hit_counter
    tracker.update([])
    tracker.update([])
    tracker.update([])  # hit_counter agotado
    ended = tracker.get_ended_tracks()
    assert len(ended) == 1
    assert ended[0]['label'] == 'person'

def test_labels_not_mixed():
    """person y chair no se mezclan aunque haya IoU de bboxes."""
    tracker = EventTracker(score_threshold=0.7, initialization_delay=0)
    r = tracker.update([
        {'label': 'person', 'bbox': (100,100,50,100), 'confidence': 0.8, 'depth_cm': None},
        {'label': 'chair', 'bbox': (100,100,60,80), 'confidence': 0.7, 'depth_cm': None},
    ])
    assert r[0]['event_id'] != r[1]['event_id']

def test_low_score_not_confirmed():
    """Detección consistente pero baja (0.52) nunca se confirma como evento."""
    tracker = EventTracker(score_threshold=0.7, initialization_delay=0)
    for _ in range(5):
        r = tracker.update([{'label': 'cell phone', 'bbox': (100,100,30,30),
                            'confidence': 0.52, 'depth_cm': None}])
    # Nunca confirma: mediana de [0.52, 0.52, 0.52...] = 0.52 < 0.70
    assert r[0]['confirmed'] == False

def test_high_score_confirms_after_padding():
    """Detección alta (0.85) se confirma cuando mediana supera threshold."""
    tracker = EventTracker(score_threshold=0.7, initialization_delay=0)
    r1 = tracker.update([{'label': 'person', 'bbox': (100,100,50,100),
                          'confidence': 0.85, 'depth_cm': None}])
    # Frame 1: mediana([0.85, 0.0, 0.0]) = 0.0 → no confirmado
    r2 = tracker.update([{'label': 'person', 'bbox': (102,101,50,100),
                          'confidence': 0.87, 'depth_cm': None}])
    # Frame 2: mediana([0.85, 0.87, 0.0]) = sorted([0.0,0.85,0.87])[1] = 0.85 → confirmado
    assert r2[0]['confirmed'] == True
    assert r2[0]['is_new_event'] == True  # justo se confirmó

def test_computed_score_is_median_padded():
    """computed_score calcula mediana con padding a 3 (patrón Frigate)."""
    from oak_vision.event_tracker import TrackMetadata
    meta = TrackMetadata(label='person', score_history=[0.85])
    # Padded: [0.85, 0.0, 0.0] → sorted: [0.0, 0.0, 0.85] → mediana = 0.0
    assert meta.computed_score == 0.0
    meta.score_history.append(0.87)
    # Padded: [0.85, 0.87, 0.0] → sorted: [0.0, 0.85, 0.87] → mediana = 0.85
    assert meta.computed_score == 0.85
```

### Test de integración (en Pi real)

```bash
#!/bin/bash
# test_integration.sh — correr en Raspberry Pi con OAK conectada
set -e

echo "=== Iniciando sistema ==="
timeout 30 python oak_coral_detector.py &
PID=$!
sleep 15

echo "=== Verificando API ==="
curl -sf http://127.0.0.1:5000/status | python -m json.tool
curl -sf http://127.0.0.1:5000/health | python -m json.tool
curl -sf http://127.0.0.1:5000/events | python -m json.tool
curl -sf http://127.0.0.1:5000/stats  | python -m json.tool

echo "=== Verificando DB ==="
sqlite3 data/oak.db "SELECT COUNT(*) FROM events;"
sqlite3 data/oak.db "SELECT COUNT(*) FROM system_events WHERE event_type='startup';"

echo "=== Verificando que events < detections ==="
EVENTS=$(sqlite3 data/oak.db "SELECT COUNT(*) FROM events;")
DETS=$(sqlite3 data/oak.db "SELECT COUNT(*) FROM detections;")
echo "Events: $EVENTS, Detections: $DETS"
[ "$EVENTS" -lt "$DETS" ] && echo "OK: eventos < detecciones" || echo "FAIL"

echo "=== Cleanup ==="
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "DONE"
```

### Checklist pre-merge

- [ ] `pip install norfair` en requirements.txt (≥2.2.0, BSD license)
- [ ] `MIN_SCORE=0.50` como default (Nivel 1, patrón Frigate)
- [ ] `SCORE_THRESHOLD=0.70` como default (Nivel 2, mediana historial)
- [ ] Norfair `initialization_delay=3` (3 frames para inicializar track)
- [ ] Norfair `hit_counter_max=15` (~2.1s tolerancia a gaps @ 7 FPS)
- [ ] TrackMetadata mantiene score_history y computed_score (mediana padded a 3)
- [ ] Solo tracks confirmados (computed_score >= threshold) generan eventos
- [ ] Cero SQL inline en api.py (todo via storage)
- [ ] Flask y stdlib usan mismas funciones
- [ ] EventTracker (wrapper Norfair) agrupa detecciones en eventos
- [ ] Tabla `events` con status active/ended
- [ ] Tabla `system_events` con telemetría
- [ ] `/events`, `/events/<id>`, `/stats`, `/health` funcionan
- [ ] OpenClaw queries.md usa tabla events por defecto
- [ ] Backoff exponencial en reconexión OAK
- [ ] Tests unitarios pasan (con `initialization_delay=0` para tests deterministas)
- [ ] Test integración en Pi pasa
- [ ] README actualizado con cambios Iter. 2

---

## Respuesta Directa: ¿Esto corrige el problema de OpenClaw?

**Sí, completamente.** La combinación de las 3 primeras fases ataca cada causa raíz:

| Causa | Fase que la corrige | Resultado |
|-------|-------------------|-----------|
| Falsos positivos (snowboard, tv, cell phone) | Fase 1: min_score 0.50 + threshold 0.70 | **Eliminados para clases observadas (score < 0.50); reducidos drásticamente en general** |
| Inflación ×frame (3,554 en vez de 1) | Fase 2: EventTracker + events | **1 evento = 1 objeto real** |
| Queries reportan frames en vez de objetos | Fase 4: queries sobre events | **OpenClaw dice "1 persona"** |

**Antes (con el código actual):**
```
OpenClaw: "En las últimas horas se detectaron 3,554 personas,
903 sillas, 752 celulares, 574 televisores..."
```

**Después (con Iteración 2):**
```
OpenClaw: "Hay 1 persona activa (lleva ~10 minutos, a 145cm)
y 1 silla. No se detectó nada más."
```

La clave es que OpenClaw dejará de hacer `COUNT(*)` sobre `detections` y pasará a consultar `events`, donde 1 fila = 1 objeto real en la escena. Los falsos positivos nunca llegan a la DB porque el threshold los elimina antes.

---

## Dependencia nueva: Norfair

```bash
# En requirements.txt (o pyproject.toml)
norfair>=2.2.0    # Kalman tracker — BSD license, ~200KB, sin deps pesadas
numpy>=1.21       # ya existente (usado por depthai y norfair)
```

Instalación en Pi 4: `pip install norfair` (sin extras). No requiere GPU, OpenCV opcional (no necesario para tracking puro). Si el proyecto usa Docker, agregar al `RUN pip install` del Dockerfile principal (NO del Coral Docker — Norfair corre en el host/orchestrator).

---

## Árbol de archivos modificados

```
oak_vision/
├── config.py           # +min_score (0.50), +score_threshold (0.70), +tracker params
├── capture.py          # sin cambios (pipeline OAK igual)
├── event_tracker.py    # NUEVO: Norfair (Kalman) tracker + two-level threshold + event lifecycle
├── inference.py        # sin cambios
├── depth.py            # sin cambios
├── storage.py          # +events schema, +create/update/close_event,
│                       #  +get_events/stats/closest, +log_system_event,
│                       #  +_query helper, +system_events schema
├── api.py              # refactor: usa storage functions, +/events, +/health
├── display.py          # sin cambios
├── hardening.py        # NUEVO: ExponentialBackoff
└── __init__.py

oak_coral_detector.py   # +tracker integration, +backoff, +telemetry calls
openclaw/queries.md     # reescrito: events por defecto, detections solo debug
openclaw/SKILL.md       # +regla "siempre usar events"

tests/
├── test_event_tracker.py  # NUEVO
├── test_storage.py        # NUEVO
├── test_api.py            # NUEVO
└── test_config.py         # NUEVO

test_integration.sh     # NUEVO: smoke test en Pi
```

---

## Roadmap: Iteración 3 (mejoras opcionales)

La Iteración 2 resuelve completamente el problema de inflación de datos (3,554 → 1 evento). La Iteración 3 se enfoca en mejoras incrementales sobre la misma arquitectura, sin cambiar el motor de inferencia (Coral USB):

| Item | Descripción |
|------|------------|
| Filtros geométricos | `min_area`, `max_area`, `min_ratio`, `max_ratio` por clase — patrón Frigate para descartar detecciones con formas imposibles |
| Zonas de exclusión | Máscaras configurables para ignorar regiones con falsos positivos recurrentes (ej: reflejos en ventana) |
| Snapshots de eventos | Guardar 1 imagen representativa por evento (frame con mayor confidence) para revisión visual |
| Multi-cámara | Extender el pipeline para soportar más de 1 OAK-D Lite simultáneamente |
| Dashboard web | Interfaz para visualizar eventos en tiempo real y revisar historial |

### Nota sobre el Coral USB

El Coral USB TPU funciona correctamente y no requiere migración. Los problemas de driver reportados en la comunidad (gasket-dkms incompatible con kernels Linux 6.x) afectan exclusivamente a modelos PCIe/M.2, no al USB. Además, nuestro Coral corre dentro de Docker (debian:bullseye) lo cual lo aísla de cambios en el sistema operativo host.

Google ha anunciado "Coral NPU" como evolución open-source del EdgeTPU, con compatibilidad de modelos existentes. Si en el futuro el Coral USB presenta algún problema de soporte, las alternativas documentadas son: (a) correr la inferencia directamente en el Myriad X de la OAK-D Lite, que soporta MobileNet-SSD y ObjectTracker nativo, o (b) migrar a Hailo-8L u OpenVINO sobre Intel iGPU, ambos soportados por Frigate.

---

## Verificación: Cobertura de los 5 Puntos del State-of-Art Review

| # | Punto del review | ¿Resuelto en el plan? | Dónde |
|---|-----------------|----------------------|-------|
| 1 | **Tracking/deduplicación** | **Sí** — Norfair (Kalman filter) + two-level threshold (patrón Frigate). Elegido sobre IoU manual (fragmenta tracks), ByteTrack (overkill para ≤5 objetos), DeepSORT (requiere 2ª red neuronal inviable en Pi 4). Licencia BSD. | Fase 2 |
| 2 | **Paridad API**: centralizar SQL en storage.py | **Sí** | Fase 3 |
| 3 | **Consultas por evento**: queries sobre events, no detections | **Sí** | Fase 4 |
| 4 | **Hardening OAK**: patrón Luxonis + X_LINK + DEPTHAI_WATCHDOG + backoff + telemetría | **Sí** | Fase 5a, 5b, 5c, 5d |
| 5 | **QA**: unit tests + integración en Pi | **Sí** | Fase 6 |

**Los 5 puntos quedan resueltos en Iteración 2.** El tracking con Norfair (Kalman filter + IoU distance) + two-level threshold es una solución profesional y probada para 1 cámara a 7 FPS. Norfair fue elegido sobre IoU manual (sin predicción → fragmenta tracks), ByteTrack (optimizado para escenas densas), y DeepSORT (requiere modelo de apariencia adicional). Licencia BSD, `pip install norfair`, overhead ~0.1ms/frame.
