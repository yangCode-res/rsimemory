#!/usr/bin/env bash
#SBATCH --job-name=memory-rsi-v02
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi/runs/slurm-%j.log
set -euo pipefail
ROOT=/file_storage01/home/juanliu/25_ymj/code/memory-policy-rsi
SGL=/file_storage01/home/juanliu/25_ymj/code/sglang-qwen38
cd "$ROOT"
LLM_CONFIG=${RSI_LLM_CONFIG:-$ROOT/configs/llm-roles.yaml}
[[ -r "$LLM_CONFIG" ]] || { echo LLM_CONFIG_MISSING; exit 1; }
PY="$SGL/venv/bin/python"
[[ ! -x "$ROOT/.venv/bin/python" ]] || PY="$ROOT/.venv/bin/python"
export PORT=7861 HOST=127.0.0.1 TP_SIZE=2
setsid "$SGL/launch.sh" > "$ROOT/runs/server-$SLURM_JOB_ID.log" 2>&1 &
PID=$!
trap 'kill -TERM -- "-$PID" 2>/dev/null || true; sleep 2; kill -KILL -- "-$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<120;i++)); do
  kill -0 "$PID" 2>/dev/null || { echo SERVER_FAILED; exit 1; }
  if curl -fs --max-time 3 http://127.0.0.1:7861/health >/dev/null; then ready=1; break; fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT; exit 1; }
SEMANTIC=()
[[ ! -d "$ROOT/models/bge-small-zh-v1.5" ]] || SEMANTIC=(--semantic-model-path "$ROOT/models/bge-small-zh-v1.5")
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
"$PY" -m rsi.cli --dataset /file_storage01/home/juanliu/25_ymj/code/MedMemoryBench \
  --output "$ROOT/runs/live-v02-$SLURM_JOB_ID" --base-url http://127.0.0.1:7861/v1 \
  --llm-config "$LLM_CONFIG" \
  --checkpoint 20 --patients-per-split 1 --per-type 1 --evolve "${SEMANTIC[@]}"
echo LIVE_PIPELINE_FINISHED
