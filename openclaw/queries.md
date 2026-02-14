# queries.md — Catálogo SQLite para Oak Vision

DB: `data/oak.db`
Tabla: `detections(timestamp, mode, infer_ms, label, confidence, bbox_x, bbox_y, bbox_w, bbox_h, depth_cm)`

---

## 1) Últimas detecciones

```sql
SELECT timestamp, label, confidence, depth_cm, mode
FROM detections
ORDER BY id DESC
LIMIT 20;
```

## 2) Conteo por etiqueta (últimas 24h)

```sql
SELECT label, COUNT(*) AS n
FROM detections
WHERE timestamp >= datetime('now', '-24 hours')
GROUP BY label
ORDER BY n DESC;
```

## 3) Objeto más cercano por etiqueta

```sql
SELECT label, MIN(depth_cm) AS closest_cm
FROM detections
WHERE depth_cm IS NOT NULL
GROUP BY label
ORDER BY closest_cm ASC;
```

## 4) Hora de mayor actividad hoy

```sql
SELECT strftime('%H:00', timestamp) AS hour, COUNT(*) AS n
FROM detections
WHERE date(timestamp)=date('now')
GROUP BY hour
ORDER BY n DESC
LIMIT 3;
```

## 5) Historial corto (últimos N minutos)

```sql
SELECT timestamp, label, confidence, depth_cm
FROM detections
WHERE timestamp >= datetime('now', '-60 minutes')
ORDER BY id DESC
LIMIT 200;
```

---

## CLI de ejemplo

```bash
python - <<'PY'
import sqlite3
conn = sqlite3.connect('data/oak.db')
for row in conn.execute("SELECT label, COUNT(*) FROM detections GROUP BY label ORDER BY 2 DESC LIMIT 5"):
    print(row)
conn.close()
PY
```
