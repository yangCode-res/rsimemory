#!/usr/bin/env bash
#SBATCH --job-name=memory-rsi-full-v03
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/full-%j.log
set -euo pipefail
umask 077
ROOT=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi
DATA=/file_storage01/home/juanliu/25_ymj/code/MedMemoryBench
PY="$ROOT/.venv/bin/python"
# Private credentials never enter the archived role YAML or CLI arguments.
LLM_ENV_FILE=${RSI_LLM_ENV_FILE:-$ROOT/.secrets/llm.env}
if [[ -r "$LLM_ENV_FILE" ]]; then
  set -a
  source "$LLM_ENV_FILE"
  set +a
fi
ARCHIVE="$ROOT/runs/full-v03-local-$SLURM_JOB_ID"
STORAGE_JSON=$("$PY" "$ROOT/scripts/round_storage.py" create --job-id "$SLURM_JOB_ID")
LOCAL=$(printf '%s' "$STORAGE_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["path"])')
echo "LOCAL_SCRATCH=$LOCAL NODE=$SLURMD_NODENAME ARCHIVE=$ARCHIVE"
ARTIFACTS="$LOCAL/artifacts"
META="$ARTIFACTS/meta"
CODE="$LOCAL/code"
OUT="$ARTIFACTS/result"
mkdir -p "$META" "$LOCAL/cache" "$LOCAL/tmp"
exec >> "$ARTIFACTS/launcher.log" 2>&1
printf '%s\n' "$STORAGE_JSON" > "$META/storage.json"
printf '%s\n' "$LOCAL" > "$META/local-root"
printf '%s\n' "$SLURMD_NODENAME" > "$META/node"
printf '%s\n' "$ARCHIVE" > "$META/archive-directory"
PID='' CLI_PID='' COPY_PID=''
MON_PIDS=()
HELPER="$ROOT/scripts/round_storage.py"
stop_group() {
  local group=$1
  local grace_s=${2:-2}
  [[ -n "$group" ]] || return 0
  kill -TERM -- "-$group" 2>/dev/null || true
  local elapsed=0 state=''
  while ((elapsed < grace_s)); do
    state=$(ps -o stat= -p "$group" 2>/dev/null | tr -d ' ' || true)
    [[ -z "$state" || "$state" == Z* ]] && break
    sleep 1
    ((elapsed += 1))
  done
  state=$(ps -o stat= -p "$group" 2>/dev/null | tr -d ' ' || true)
  if [[ -n "$state" && "$state" != Z* ]]; then
    kill -KILL -- "-$group" 2>/dev/null || true
  fi
  wait "$group" 2>/dev/null || true
}
cleanup() {
  local result=$?
  trap - EXIT INT TERM
  for monitor_pid in "${MON_PIDS[@]}"; do stop_group "$monitor_pid"; done
  [[ -z "$COPY_PID" ]] || { kill "$COPY_PID" 2>/dev/null || true; wait "$COPY_PID" 2>/dev/null || true; }
  stop_group "$CLI_PID"
  # Give the Python supervisor time to terminate and reap every server process
  # group before the final artifact archive checksums any logs.
  stop_group "$PID" 25
  printf '%s\n' "$result" > "$META/exit-code"
  if [[ "$result" == 0 ]]; then echo completed > "$META/status"; else echo failed > "$META/status"; fi
  echo FINAL_ARCHIVE_START
  if ! "$PY" "$HELPER" finalize --source "$ARTIFACTS" --destination "$ARCHIVE" > "$LOCAL/final-archive.json" 2> "$LOCAL/final-archive.error"; then
    echo archive_failed > "$META/status"
    printf '%s\n' 74 > "$META/exit-code"
    echo "ARCHIVE_FAILED_LOCAL_RETAINED=$LOCAL"
    exit 74
  fi
  # No writes to artifact logs after the final checksum verification.
  exit "$result"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo preflight > "$META/status"
COMMIT=${RSI_IMPLEMENTATION_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}
git clone --quiet --no-hardlinks "$ROOT" "$CODE"
git -C "$CODE" checkout --quiet --detach "$COMMIT"
HELPER="$CODE/scripts/round_storage.py"
git -C "$CODE" rev-parse HEAD > "$META/implementation-commit"
cd "$CODE"
# Freeze the selected role YAML before model startup. New tasks always use it.
LLM_CONFIG_SOURCE=${RSI_LLM_CONFIG:-$CODE/configs/llm-roles.yaml}
[[ -r "$LLM_CONFIG_SOURCE" ]] || { echo LLM_CONFIG_MISSING; exit 1; }
cp "$LLM_CONFIG_SOURCE" "$META/llm-roles.yaml"
printf '%s\n' "$LLM_CONFIG_SOURCE" > "$META/llm-config-source"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR="$LOCAL/tmp" XDG_CACHE_HOME="$LOCAL/cache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/cache/torch" TRITON_CACHE_DIR="$LOCAL/cache/triton"
export TORCH_EXTENSIONS_DIR="$LOCAL/cache/torch_extensions" CUDA_CACHE_PATH="$LOCAL/cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$LOCAL/cache/flashinfer" SGLANG_CACHE_DIR="$LOCAL/cache/sglang"
export SGLANG_DG_CACHE_DIR="$LOCAL/cache/deep_gemm" CUTE_DSL_CACHE_DIR="$LOCAL/cache/cute"
export HF_HOME="$LOCAL/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" scripts/preflight_v03.py --dataset "$DATA" > "$META/preflight.json"
echo PREFLIGHT_OK
"$PY" scripts/preflight_external_llm.py --config "$META/llm-roles.yaml" > "$META/external-llm-preflight.json"
echo EXTERNAL_LLM_PREFLIGHT_OK
(
  trap 'exit 0' TERM INT
  while true; do
    if "$PY" "$HELPER" sync --source "$ARTIFACTS" --destination "$ARCHIVE" > "$LOCAL/last-sync.json" 2> "$LOCAL/last-sync.error"; then
      echo ok > "$META/last-sync-status"
    else echo degraded > "$META/last-sync-status"; fi
    sleep 60 & wait $! || exit 0
  done
) &
COPY_PID=$!
echo model_starting > "$META/status"
# The diagnostic input is never clipped by the application. Use the model's
# verified native context capacity; request budgets come from the frozen YAML.
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
printf '%s\n' "$CONTEXT_LENGTH" > "$META/service-context-length"
setsid "$PY" "$CODE/scripts/launch_sglang_replicas.py" \
  --root "$CODE" --local-root "$LOCAL" --meta-dir "$META" \
  --startup-timeout-s 1800 > "$ARTIFACTS/replica-supervisor.log" 2>&1 &
PID=$!
ready=0
startup_deadline=$((SECONDS + 1900))
while ((SECONDS < startup_deadline)); do
  kill -0 "$PID" 2>/dev/null || { echo SERVER_FAILED; exit 1; }
  if [[ -s "$META/replicas-ready.json" ]]; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT; exit 1; }
echo replay_running > "$META/status"
echo MODEL_READY_FULL_REPLAY_START
RESUME_ARGS=()
if [[ -n "${RSI_RESUME_BASELINES_FROM:-}" ]]; then
  RESUME_ARGS=(--resume-baselines-from "$RSI_RESUME_BASELINES_FROM")
  printf '%s\n' "$RSI_RESUME_BASELINES_FROM" > "$META/baseline-resume-source"
fi
LLM_ARGS=(--llm-config "$META/llm-roles.yaml")
SEARCH_ARGS=()
if [[ "${RSI_CONTINUOUS_SEARCH:-0}" == 1 ]]; then
  SEARCH_ARGS=(--search-profile "$CODE/deploy/evolution-search.json")
fi
setsid "$PY" -m rsi.cli --dataset "$DATA" --output "$OUT" --language zh \
  --base-url http://127.0.0.1:7862/v1 \
  --candidate-base-urls http://127.0.0.1:7862/v1 http://127.0.0.1:7863/v1 http://127.0.0.1:7864/v1 \
  --candidate-workers 3 \
  --reflexion-max-corrections "${RSI_REFLEXION_MAX_CORRECTIONS:-1}" \
  "${RESUME_ARGS[@]}" "${SEARCH_ARGS[@]}" "${LLM_ARGS[@]}" \
  --semantic-model-path "$ROOT/models/bge-small-zh-v1.5" --evolve &
CLI_PID=$!
while IFS=$'\t' read -r server_pid health_url service_meta; do
  [[ -n "$server_pid" ]] || continue
  setsid "$PY" "$CODE/scripts/service_monitor.py" \
    --server-pid "$server_pid" --client-pid "$CLI_PID" \
    --meta-dir "$service_meta" --url "$health_url" \
    --timeout-s 30 --interval-s 10 --threshold 3 --grace-s 120 &
  MON_PIDS+=("$!")
done < <("$PY" - "$META/replicas.json" <<'PY'
import json, sys
for replica in json.load(open(sys.argv[1]))["replicas"]:
    print(f'{replica["pid"]}\t{replica["health_url"]}\t{replica["meta_dir"]}')
PY
)
[[ "${#MON_PIDS[@]}" == 3 ]] || { echo REPLICA_MONITOR_SETUP_FAILED; exit 1; }
set +e
wait "$CLI_PID"
result=$?
set -e
if [[ "$result" == 0 ]]; then
  supervisor_state=$(ps -o stat= -p "$PID" 2>/dev/null | tr -d ' ' || true)
  if [[ -z "$supervisor_state" || "$supervisor_state" == Z* ]]; then
    echo REPLICA_SUPERVISOR_EXITED
    result=1
  fi
fi
if [[ "$result" == 0 ]]; then echo FULL_ROUND_FINISHED; fi
exit "$result"
