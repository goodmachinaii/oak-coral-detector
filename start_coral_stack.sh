#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Levanta backend Coral en Docker.
# Por defecto NO reconstruye imagen para evitar reinicios USB innecesarios.
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  sudo docker compose up -d --build
else
  sudo docker compose up -d
fi

# Espera a que la API Coral responda saludable antes de abrir la app
for i in {1..20}; do
  if curl -fsS "http://127.0.0.1:8765/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# Limpia flag y arranca app host
rm -f STOP_OAK_CORAL_DETECTOR.flag
VENV_ACTIVATE="${OAK_CORAL_VENV_ACTIVATE:-$PWD/.venv/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
exec python oak_coral_detector.py
