# SKILL.md — Oak Vision Queries

Usa este skill cuando el usuario pregunte por **qué detectó la cámara**, estadísticas, historial o distancia de objetos.

## Fuentes de datos

1. **Preferido:** API local
- Base URL: `http://127.0.0.1:5000`
- Endpoints:
  - `GET /status`
  - `GET /detections`
  - `GET /detections/stats?hours=24`

2. **Fallback:** SQLite local
- DB: `data/oak.db`
- Tabla principal: `detections`

## Reglas

- Consultar solo cuando el usuario lo pida.
- Responder en lenguaje natural, breve y con unidades claras (cm/m).
- Si no hay datos recientes, decirlo explícitamente.
- Si API no responde, usar queries SQLite del catálogo `queries.md`.

## Ejemplos de intención

- "¿Qué ves ahora?"
- "¿Qué detectaste en la última hora?"
- "¿Cuál fue el objeto más cercano?"
- "¿A qué hora hubo más movimiento?"
