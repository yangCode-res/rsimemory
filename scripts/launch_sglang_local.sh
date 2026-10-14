#!/usr/bin/env bash
# Preserve model/runtime settings; the job supplies node-local caches.
set -euo pipefail
SGL=/file_storage01/home/juanliu/25_ymj/code/sglang-qwen38
: "${TORCHINDUCTOR_CACHE_DIR:?local cache required}"
: "${TRITON_CACHE_DIR:?local cache required}"
export PATH="$SGL/venv/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
"$SGL/venv/bin/python" -c 'import torch; assert torch.cuda.is_available(), "GPU allocation is required"; print(torch.cuda.get_device_name(), torch.cuda.device_count())'
ARGS=(
 --model-path /file_storage01/home/juanliu/25_ymj/memory/memory/vendor/openresearcher/models/Qwen3.8-27B
 --served-model-name qwen3.8-27b
 --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}"
 --tensor-parallel-size "${TP_SIZE:-2}" --dtype bfloat16
 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
 --max-running-requests 6 --context-length "${CONTEXT_LENGTH:-32768}"
 --mem-fraction-static "${MEM_FRACTION:-0.85}"
 --chunked-prefill-size 2048 --schedule-policy lpm --stream-interval 1
 --mamba-full-memory-ratio 0.1
)
[[ -z "${ATTENTION_BACKEND:-}" ]] || ARGS+=(--attention-backend "$ATTENTION_BACKEND")
[[ -z "${MAMBA_BACKEND:-}" ]] || ARGS+=(--mamba-backend "$MAMBA_BACKEND")
[[ -z "${KV_CACHE_DTYPE:-}" ]] || ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -z "${API_KEY:-}" ]] || ARGS+=(--api-key "$API_KEY")
exec "$SGL/venv/bin/python" -m sglang.launch_server "${ARGS[@]}" "$@"
