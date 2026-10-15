# Shared helper for serving SEA-LION models via vLLM on the devbox.
# SOURCE this; do not execute directly. See docs/reproducing.md for the full story.
#
# Provides:
#   launch_vllm MODEL SERVED_NAME PORT [extra vllm serve args...]
# Env overrides: GPU=N (skip auto-pick), MAX_MODEL_LEN, GPU_MEM_UTIL, VLLM_BIN, CUDA_HOME.

VLLM_BIN="${VLLM_BIN:-vllm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_setup_cuda_env() {
  # cu129 toolkit is required so FlashInfer's runtime JIT can link -lcudart (SEA-LION v4.5).
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
  export PATH="$CUDA_HOME/bin:$PATH"
  # IMPORTANT: set LD_LIBRARY_PATH to ONLY the cu129 lib. Do NOT inherit the devbox
  # default (EFA/NCCL + generic /usr/local/cuda/lib) — it shadows torch's bundled CUDA
  # runtime and makes torch.cuda.is_available() fail with "CUDA driver initialization
  # failed", even though nvidia-smi and raw cuInit work fine.
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
  if [ -f "$REPO_DIR/.env" ]; then
    export HF_TOKEN="$(grep -E '^HF_TOKEN=' "$REPO_DIR/.env" | cut -d= -f2-)"
  fi
}

# Print the first idle GPU index (<1 GiB used). Honors GPU= override.
# No early `exit`/`head` in the pipeline: closing it early sends SIGPIPE upstream,
# which `set -o pipefail` turns into a 141 that aborts the caller.
_pick_gpu() {
  if [ -n "${GPU:-}" ]; then echo "$GPU"; return; fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' 'picked {next} $2+0 < 1000 {gsub(/ /,"",$1); print $1; picked=1}'
}

# launch_vllm MODEL SERVED_NAME PORT [extra vllm serve args...]
# Replaces the shell with `vllm serve` (exec), so the calling script becomes the server.
launch_vllm() {
  local model="$1" served="$2" port="$3"; shift 3
  _setup_cuda_env
  local gpu; gpu="$(_pick_gpu)"
  if [ -z "$gpu" ]; then echo "No idle GPU found (all >1GiB used). Set GPU=N explicitly." >&2; return 1; fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "Serving $model"
  echo "  served-model-name=$served  GPU=$gpu  PORT=$port  CUDA_HOME=$CUDA_HOME"
  exec "$VLLM_BIN" serve "$model" \
    --served-model-name "$served" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --port "$port" \
    "$@"
}
