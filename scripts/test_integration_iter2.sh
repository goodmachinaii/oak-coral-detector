#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  ./stop_coral_stack.sh >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[1/5] Starting stack (HEADLESS=1)..."
HEADLESS=1 ./start_coral_stack.sh >/tmp/oak_iter2_integration.log 2>&1 &
PID=$!
sleep 10

echo "[2/5] Checking /status"
curl -fsS http://127.0.0.1:5000/status >/tmp/oak_status.json

echo "[3/5] Checking /health"
curl -fsS http://127.0.0.1:5000/health >/tmp/oak_health.json

echo "[4/5] Checking /events and /stats"
curl -fsS "http://127.0.0.1:5000/events?hours=1" >/tmp/oak_events.json
curl -fsS "http://127.0.0.1:5000/stats?hours=1" >/tmp/oak_stats.json

echo "[5/5] Validating DB schema"
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect('data/oak.db')
cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('events','system_events')")
found = {r[0] for r in cur.fetchall()}
assert 'events' in found and 'system_events' in found, f"Missing tables: {found}"
print('Schema OK')
PY

kill "$PID" >/dev/null 2>&1 || true
wait "$PID" >/dev/null 2>&1 || true

echo "Integration check OK"
