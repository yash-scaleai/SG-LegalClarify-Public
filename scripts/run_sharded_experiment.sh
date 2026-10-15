#!/usr/bin/env bash
#
# Launch a sharded run of run_eval.py in parallel. Each shard gets its own log
# file and its own CoT sidecar so nothing collides.
#
# Usage:
#   scripts/run_sharded_experiment.sh [-- ANY_RUN_EVAL_FLAGS ...]
#
# Environment overrides (all optional):
#   N_SHARDS         number of parallel shards (default 10)
#   PYTHON           python binary (default: python3 from .venv/bin if present)
#   RUN_TAG          human tag included in log/CoT filenames (default: UTC time)
#
# Examples:
#   # Default 10 shards, whatever models/prompts are in run_eval's defaults:
#   scripts/run_sharded_experiment.sh
#
#   # Restrict to two models, allow overwriting existing outputs:
#   scripts/run_sharded_experiment.sh -- \
#     --models anthropic/claude-opus-4-8 fireworks_ai/deepseek-v4-flash \
#     --allow-overwrite
#
#   # 5 shards instead of 10:
#   N_SHARDS=5 scripts/run_sharded_experiment.sh
#
# Notes:
#   - Watch progress live with: tail -f logs/shard_*_${RUN_TAG}.log
#   - Rate limits: 10 concurrent processes multiply the LiteLLM proxy's TPM by
#     ~10x. If you hit 429s, drop N_SHARDS to 4-6.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- config ---
N_SHARDS="${N_SHARDS:-10}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x ".venv/bin/python3" ]]; then
  PY=".venv/bin/python3"
else
  PY="python3"
fi

# Pass-through: everything after `--` goes to run_eval.py verbatim.
USER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  USER_ARGS=("$@")
fi

mkdir -p logs data

# --- Preflight: probe every model with a trivial request before we commit to a
# multi-hour run. Any provider-side 400 (unsupported reasoning_effort, max_tokens
# ceiling, missing model access, etc.) surfaces here in ~60 seconds instead of
# after hours of wasted trials. Set SKIP_PREFLIGHT=1 to bypass.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "=== preflight: probing every model with reasoning enabled ==="
  # Forward the --models flag from user args if present so we only probe the
  # models the run will actually exercise.
  probe_args=()
  for ((i=0; i<${#USER_ARGS[@]}; i++)); do
    if [[ "${USER_ARGS[$i]}" == "--models" ]]; then
      j=$((i+1))
      while [[ $j -lt ${#USER_ARGS[@]} && "${USER_ARGS[$j]:0:2}" != "--" ]]; do
        probe_args+=("${USER_ARGS[$j]}")
        j=$((j+1))
      done
      break
    fi
  done
  if (( ${#probe_args[@]} > 0 )); then
    "$PY" scripts/preflight_models.py --models "${probe_args[@]}"
  else
    "$PY" scripts/preflight_models.py
  fi
  pf_rc=$?
  if (( pf_rc != 0 )); then
    echo "[preflight] failed; aborting launch. Fix the reported issues or "
    echo "            re-run with SKIP_PREFLIGHT=1 to bypass at your own risk."
    exit "$pf_rc"
  fi
  echo
fi

echo "=== sharded experiment launch ==="
echo "  python:      $PY"
echo "  shards:      $N_SHARDS"
echo "  run tag:     $RUN_TAG"
echo "  user args:   ${USER_ARGS[*]:-<none>}"
echo "  logs:        logs/shard_<i>_of_${N_SHARDS}_${RUN_TAG}.log"
echo "  cot files:   data/experiment_cot_${RUN_TAG}_shard_<i>.jsonl"
echo "==============================="
echo

# --- launch shards ---
pids=()
for i in $(seq 1 "$N_SHARDS"); do
  log="logs/shard_${i}_of_${N_SHARDS}_${RUN_TAG}.log"
  cot="data/experiment_cot_${RUN_TAG}_shard_${i}.jsonl"
  (
    export EXPERIMENT_COT_JSONL_PATH="$cot"
    "$PY" -u run_eval.py ${USER_ARGS[@]+"${USER_ARGS[@]}"} \
      --num-shards "$N_SHARDS" --shard-index "$i" \
      > "$log" 2>&1
  ) &
  pid=$!
  pids+=("$pid")
  echo "[launch] shard $i/$N_SHARDS  pid=$pid  log=$log"
done

echo
echo "[wait] all $N_SHARDS shards running; blocking until they finish..."
echo "       tail -f logs/shard_*_${RUN_TAG}.log  (in another terminal)"

# --- wait for all shards, tracking failures ---
failed_shards=()
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  shard_num=$((i + 1))
  if wait "$pid"; then
    echo "[ok]  shard $shard_num/$N_SHARDS (pid $pid) exit 0"
  else
    rc=$?
    echo "[ERR] shard $shard_num/$N_SHARDS (pid $pid) exit $rc"
    failed_shards+=("$shard_num")
  fi
done

if (( ${#failed_shards[@]} > 0 )); then
  echo
  echo "===================================================================="
  echo "STOP: ${#failed_shards[@]} shard(s) failed: ${failed_shards[*]}"
  echo "Check logs/shard_*_${RUN_TAG}.log for errors. After fixing, resume with"
  echo "a new RUN_TAG and --new-data-only so finished scenarios are skipped."
  echo "===================================================================="
  exit 1
fi

echo
echo "=== done ==="
echo "logs:      logs/shard_*_${RUN_TAG}.log"
echo "cot files: data/experiment_cot_${RUN_TAG}_shard_*.jsonl"
