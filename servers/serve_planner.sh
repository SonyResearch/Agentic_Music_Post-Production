#!/usr/bin/env bash
# Serve the Qwen3-Omni-30B-A3B-Thinking planner with vLLM (OpenAI-compatible API).
#
#   bash servers/serve_planner.sh                      # GPU visible to the process, port 8000
#   PORT=8001 CUDA_VISIBLE_DEVICES=1 bash servers/serve_planner.sh
#
# Clients: --planner-urls http://127.0.0.1:8000/v1 (comma-separate several servers to run
# songs in parallel). --max-num-seqs only changes throughput, not outputs. Extra
# arguments are passed through to `vllm serve`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-${MIXAGENT_MODELS:-$REPO/models}/Qwen3-Omni-30B-A3B-Thinking}"
SERVED_NAME="${SERVED_NAME:-Qwen3-Omni-30B-A3B-Thinking}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
unset VLLM_PORT  # reserved by vLLM for its internal port; never let it leak in

# Newer vLLM builds can enforce the client's thinking_token_budget via --reasoning-config.
REASONING_ARGS=(--reasoning-parser qwen3)
if vllm serve --help=all 2>/dev/null | grep -q -- '--reasoning-config'; then
  REASONING_ARGS+=(--reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"\n\nConsidering the token budget, I will now provide the final answer.</think>\n\n"}')
fi

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.90 \
  "${REASONING_ARGS[@]}" \
  "$@"
