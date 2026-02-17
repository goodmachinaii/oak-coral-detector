# queries.md — Catálogo SQL para Oak Vision

DB: `data/oak.db`

## Eventos (usar por defecto)

```sql
SELECT label, COUNT(*) AS event_count, SUM(duration_sec) AS total_seconds
FROM events
WHERE first_seen >= datetime('now', '-24 hours')
GROUP BY label
ORDER BY event_count DESC;
```

```sql
SELECT label, event_id, first_seen, max_confidence, min_depth_cm
FROM events
WHERE status='active'
ORDER BY first_seen DESC;
```

```sql
SELECT * FROM events
WHERE event_id = ?;
```

## Legacy/debug (detecciones por frame)

```sql
SELECT label, COUNT(*) AS raw_frames
FROM detections
WHERE timestamp >= datetime('now', '-24 hours')
GROUP BY label
ORDER BY raw_frames DESC;
```
