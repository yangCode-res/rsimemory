#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$HERE/../.." && pwd)"
DATA_ROOT="${LANGFUSE_DATA_ROOT:-$PROJECT_ROOT/observability/langfuse}"
ENV_FILE="$DATA_ROOT/.env"
COMPOSE_FILE="$HERE/compose.yaml"

prepare() {
  mkdir -p "$DATA_ROOT" "$DATA_ROOT/data"/{postgres,clickhouse,clickhouse-logs,minio,redis} "$DATA_ROOT/state" "$DATA_ROOT/logs"
  chmod 700 "$DATA_ROOT" "$DATA_ROOT/data" "$DATA_ROOT/data"/* "$DATA_ROOT/state" "$DATA_ROOT/logs"
  if [[ ! -e "$ENV_FILE" ]]; then
    python3 - "$ENV_FILE" <<'PY'
import os
import secrets
import sys

env_path = sys.argv[1]
values = {
    "NEXTAUTH_URL": "http://localhost:3000",
    "POSTGRES_PASSWORD": secrets.token_hex(32),
    "CLICKHOUSE_PASSWORD": secrets.token_hex(32),
    "REDIS_AUTH": secrets.token_hex(32),
    "MINIO_ROOT_USER": "langfuse",
    "MINIO_ROOT_PASSWORD": secrets.token_hex(32),
    "SALT": secrets.token_hex(32),
    "ENCRYPTION_KEY": secrets.token_hex(32),
    "NEXTAUTH_SECRET": secrets.token_hex(32),
    "LANGFUSE_INIT_ORG_ID": "rsi-memory",
    "LANGFUSE_INIT_ORG_NAME": "RSI Memory",
    "LANGFUSE_INIT_PROJECT_ID": "memory-policy-rsi",
    "LANGFUSE_INIT_PROJECT_NAME": "Memory Policy RSI",
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": "pk-lf-" + secrets.token_hex(20),
    "LANGFUSE_INIT_PROJECT_SECRET_KEY": "sk-lf-" + secrets.token_hex(32),
    "LANGFUSE_INIT_USER_EMAIL": "rsi-admin@rsi.local",
    "LANGFUSE_INIT_USER_NAME": "RSI Observability Admin",
    "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(32),
}
os.umask(0o077)
with open(env_path, "x", encoding="utf-8") as f:
    for key, value in values.items():
        f.write(f"{key}={value}\n")
os.chmod(env_path, 0o600)
PY
    echo "Created private Langfuse credentials in $ENV_FILE"
  else
    chmod 600 "$ENV_FILE"
    echo "Using existing private Langfuse credentials in $ENV_FILE"
  fi

}

compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
  elif command -v podman-compose >/dev/null 2>&1; then
    podman-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
  elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
    podman compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
  else
    echo "Install Docker Compose or Podman with a Compose provider before starting Langfuse." >&2
    return 1
  fi
}

action="${1:-prepare}"
case "$action" in
  prepare)
    prepare
    ;;
  up|down|ps|logs|pull)
    prepare
    compose "$action" "${@:2}"
    ;;
  config)
    prepare
    compose config >/dev/null
    echo "Compose configuration is valid."
    ;;
  *)
    echo "Usage: $0 {prepare|config|pull|up|ps|logs|down}" >&2
    exit 2
    ;;
esac
