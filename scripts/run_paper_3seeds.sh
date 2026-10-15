#!/usr/bin/env bash
#
# Launch the paper's 3-seed evaluation sweep.
#
# For each seed S in {42, 43, 44}, this script:
#   1. Sets --results-root  = results_paper_run_<i>/
#   2. Sets --seed          = S              (threaded into every reasoning-enabled primary call)
#   3. Sets --run-id        = i              (stamped in every CoT record)
#   4. Writes CoT sidecars to data/experiment_cot_<RUN_BASENAME>_<i>_seed<S>_shard_<j>.jsonl
#   5. Invokes scripts/run_sharded_experiment.sh, which fans out to N_SHARDS
#      parallel workers, runs preflight, and blocks until all shards finish.
#
# The three runs execute SEQUENTIALLY (not in parallel) so the LiteLLM proxy's
# TPM budget is not multiplied 3× on top of the intra-run sharding.
#
# The simulator and classifier do NOT receive the seed. They present the same
# conversation across all three runs, so the 3 samples measure model response
# variance conditional on identical interlocutors.
#
# Usage:
#   scripts/run_paper_3seeds.sh                        # 3 seeds, default 10 shards each
#   N_SHARDS=6 scripts/run_paper_3seeds.sh             # each seed uses 6 shards
#   SEEDS="42 43 44 45" scripts/run_paper_3seeds.sh    # override seed set
#   RUN_BASENAME=my_run scripts/run_paper_3seeds.sh    # customize output roots
#   scripts/run_paper_3seeds.sh -- --allow-overwrite   # pass-through flags after --
#
# Environment overrides:
#   SEEDS         space-separated integers (default "42 43 44")
#   N_SHARDS      shards per run (default 10)
#   RUN_BASENAME  results dir stem (default "paper_run"; produces results_paper_run_1/…)
#   PYTHON        python binary (default: .venv/bin/python3 if present, else python3)
#
# Notes:
#   - This script does NOT resume from prior partial runs. If a shard failed
#     mid-way, re-run the inner sharded script directly with the same
#     RESULTS_ROOT and SEED.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${SEEDS:-42 43 44}"
N_SHARDS="${N_SHARDS:-10}"
RUN_BASENAME="${RUN_BASENAME:-paper_run}"

# Pass-through: everything after `--` goes to run_sharded_experiment.sh (and
# hence to run_eval.py) verbatim.
USER_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  USER_ARGS=("$@")
fi

echo "==================================================================="
echo "  3-seed paper evaluation sweep"
echo "==================================================================="
echo "  seeds:         $SEEDS"
echo "  shards/run:    $N_SHARDS"
echo "  output roots:  results_${RUN_BASENAME}_<i>/"
echo "  pass-through:  ${USER_ARGS[*]:-<none>}"
echo "==================================================================="
echo

run_index=0
for seed in $SEEDS; do
  run_index=$((run_index + 1))
  results_root="results_${RUN_BASENAME}_${run_index}"
  run_tag="${RUN_BASENAME}_${run_index}_seed${seed}"

  echo
  echo "-------------------------------------------------------------------"
  echo "  Run $run_index / seed=$seed"
  echo "  results_root = $results_root"
  echo "  run_tag      = $run_tag"
  echo "-------------------------------------------------------------------"
  echo

  # NOTE: run_sharded_experiment.sh reads N_SHARDS + RUN_TAG from env, and
  # forwards everything after `--` to run_eval.py. We inject --results-root,
  # --seed, --run-id, and any user pass-through args here.
  N_SHARDS="$N_SHARDS" RUN_TAG="$run_tag" \
    scripts/run_sharded_experiment.sh -- \
      --results-root "$results_root" \
      --seed "$seed" \
      --run-id "$run_index" \
      ${USER_ARGS[@]+"${USER_ARGS[@]}"}

  rc=$?
  if (( rc != 0 )); then
    echo "[ERR] run $run_index (seed=$seed) failed with exit $rc; aborting."
    exit "$rc"
  fi
  echo "[ok]  run $run_index (seed=$seed) complete."
done

echo
echo "==================================================================="
echo "  All ${run_index} seeded runs completed successfully."
echo "  Result trees: results_${RUN_BASENAME}_1/  results_${RUN_BASENAME}_2/  results_${RUN_BASENAME}_3/"
echo "  CoT sidecars: data/experiment_cot_${RUN_BASENAME}_*_shard_*.jsonl"
echo "==================================================================="
