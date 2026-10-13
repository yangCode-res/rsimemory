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
ROOT=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi
SGL=/file_storage01/home/juanliu/25_ymj/code/sglang-qwen38
DATA=/file_storage01/home/juanliu/25_ymj/code/MedMemoryBench
PY="$ROOT/.venv/bin/python"
CODE="$ROOT/runs/code-$SLURM_JOB_ID"
OUT="$ROOT/runs/full-v03-splitv2-$SLURM_JOB_ID"
META="$ROOT/runs/full-job-$SLURM_JOB_ID"
mkdir -p "$META"
PID=''
cleanup() {
  local result=$?
  trap - EXIT
  printf '%s\n' "$result" > "$META/exit-code"
  if [[ -n "$PID" ]]; then
    kill -TERM -- "-$PID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  if [[ "$result" == 0 ]]; then echo completed > "$META/status"; else echo failed > "$META/status"; fi
  exit "$result"
}
trap cleanup EXIT
echo preflight > "$META/status"
# Freeze source before a long replay; future edits to ROOT cannot affect it.
COMMIT=${RSI_IMPLEMENTATION_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}
git clone --quiet --no-hardlinks "$ROOT" "$CODE"
git -C "$CODE" checkout --quiet --detach "$COMMIT"
git -C "$CODE" rev-parse HEAD > "$META/implementation-commit"
printf '%s\n' "$OUT" > "$META/output-directory"
printf '%s\n' "$SLURMD_NODENAME" > "$META/node"
cd "$CODE"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
"$PY" scripts/preflight_v03.py --dataset "$DATA" > "$META/preflight.json"
echo PREFLIGHT_OK
echo model_starting > "$META/status"
export PORT=7862 HOST=127.0.0.1 TP_SIZE=2
setsid "$SGL/launch.sh" > "$ROOT/runs/server-full-$SLURM_JOB_ID.log" 2>&1 &
PID=$!
ready=0
for ((i=0;i<120;i++)); do
  kill -0 "$PID" 2>/dev/null || { echo SERVER_FAILED; exit 1; }
  if curl -fs --max-time 3 http://127.0.0.1:7862/health >/dev/null; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT; exit 1; }
echo replay_running > "$META/status"
echo MODEL_READY_FULL_REPLAY_START
"$PY" -m rsi.cli --dataset "$DATA" --output "$OUT" --language zh \
  --base-url http://127.0.0.1:7862/v1 \
  --semantic-model-path "$ROOT/models/bge-small-zh-v1.5" --evolve
echo FULL_ROUND_FINISHED
