#!/usr/bin/env bash
#SBATCH --job-name=memory-rsi-full-v03
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/full-%j.log
set -euo pipefail
umask 077
ROOT=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi
DATA=/file_storage01/home/juanliu/25_ymj/code/MedMemoryBench
PY="$ROOT/.venv/bin/python"
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
PID='' CLI_PID='' MON_PID='' COPY_PID=''
HELPER="$ROOT/scripts/round_storage.py"
stop_group() {
  local group=$1
  [[ -n "$group" ]] || return 0
  kill -TERM -- "-$group" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$group" 2>/dev/null || true
  wait "$group" 2>/dev/null || true
}
cleanup() {
  local result=$?
  trap - EXIT INT TERM
  stop_group "$MON_PID"
  [[ -z "$COPY_PID" ]] || { kill "$COPY_PID" 2>/dev/null || true; wait "$COPY_PID" 2>/dev/null || true; }
  stop_group "$CLI_PID"
  stop_group "$PID"
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
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR="$LOCAL/tmp" XDG_CACHE_HOME="$LOCAL/cache"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL/cache/torch" TRITON_CACHE_DIR="$LOCAL/cache/triton"
export TORCH_EXTENSIONS_DIR="$LOCAL/cache/torch_extensions" CUDA_CACHE_PATH="$LOCAL/cache/cuda"
export FLASHINFER_WORKSPACE_BASE="$LOCAL/cache/flashinfer" SGLANG_CACHE_DIR="$LOCAL/cache/sglang"
export SGLANG_DG_CACHE_DIR="$LOCAL/cache/deep_gemm" CUTE_DSL_CACHE_DIR="$LOCAL/cache/cute"
export HF_HOME="$LOCAL/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" scripts/preflight_v03.py --dataset "$DATA" > "$META/preflight.json"
echo PREFLIGHT_OK
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
export PORT=7862 HOST=127.0.0.1 TP_SIZE=2
# The diagnostic input is never clipped by the application. Use the model's
# verified native context capacity; actor/judge request budgets stay frozen.
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
printf '%s\n' "$CONTEXT_LENGTH" > "$META/service-context-length"
setsid bash "$CODE/scripts/launch_sglang_local.sh" > "$ARTIFACTS/server.log" 2>&1 &
PID=$!
ready=0
startup_deadline=$((SECONDS + 1800))
while ((SECONDS < startup_deadline)); do
  kill -0 "$PID" 2>/dev/null || { echo SERVER_FAILED; exit 1; }
  if curl -fsS --connect-timeout 3 --max-time 30 http://127.0.0.1:7862/health >/dev/null; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT; exit 1; }
echo replay_running > "$META/status"
echo MODEL_READY_FULL_REPLAY_START
setsid "$PY" -m rsi.cli --dataset "$DATA" --output "$OUT" --language zh \
  --base-url http://127.0.0.1:7862/v1 \
  --reflexion-max-tokens "${RSI_REFLEXION_MAX_TOKENS:-8192}" \
  --codegen-max-tokens "${RSI_CODEGEN_MAX_TOKENS:-16384}" \
  --reflexion-temperature "${RSI_REFLEXION_TEMPERATURE:-0.7}" \
  --codegen-temperature "${RSI_CODEGEN_TEMPERATURE:-0.4}" \
  --optimizer-timeout-s "${RSI_OPTIMIZER_TIMEOUT_S:-900}" \
  --semantic-model-path "$ROOT/models/bge-small-zh-v1.5" --evolve &
CLI_PID=$!
setsid "$PY" "$CODE/scripts/service_monitor.py" \
  --server-pid "$PID" --client-pid "$CLI_PID" --meta-dir "$META" \
  --timeout-s 30 --interval-s 10 --threshold 3 --grace-s 120 &
MON_PID=$!
set +e
wait "$CLI_PID"
result=$?
set -e
if [[ "$result" == 0 ]]; then echo FULL_ROUND_FINISHED; fi
exit "$result"
