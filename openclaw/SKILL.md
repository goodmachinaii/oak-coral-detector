# SKILL.md — Oak Vision Queries

Usa este skill cuando el usuario pregunte por detecciones de cámara, actividad o distancias.

## Regla fundamental

**SIEMPRE usar `events` para reportes al usuario.**
- `detections` cuenta frames (debug)
- `events` cuenta objetos reales (reportes)

## Fuentes

1. API local (preferida): `http://127.0.0.1:5000`
   - `GET /status`
   - `GET /latest`
   - `GET /events`
   - `GET /events/<id>`
   - `GET /stats`
   - `GET /health`

2. SQLite fallback: `data/oak.db`
   - Tabla de reportes: `events`
   - Tabla raw/debug: `detections`

## Reglas

- Consultar solo cuando el usuario lo pida.
- Responder breve y en lenguaje natural.
- Timestamps están en UTC; al responder, convertir/explicar hora local si aplica.
- Si no hay datos recientes, decirlo explícitamente.
