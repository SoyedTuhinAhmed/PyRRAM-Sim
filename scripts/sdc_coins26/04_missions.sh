#!/usr/bin/env bash
# Step 4: fly the simulated chips through the 12-month mission, for every configuration of Figs. 2-8.
# One job = one configuration x one seed, run one per GPU (longest first); finished jobs are kept, so an
# interrupted run continues where it stopped.
#
#   bash scripts/sdc_coins26/04_missions.sh                    # default seed plan, ~17 GPU-hours
#   bash scripts/sdc_coins26/04_missions.sh --full             # 10 seeds everywhere, ~42 GPU-hours
#   bash scripts/sdc_coins26/04_missions.sh --figures "fig2 fig3" --seeds 0,1,2
#
# Default seed plan: 10 seeds for Figs. 2-4 (as in the paper), 5 for Fig. 5, 2 for Figs. 6-8.
# EVAL_BATCH=16 lowers the GPU memory per job (default 39 images = 40,920 patches).
# --fused adds a cheaper recalibration backward to torch.compile (faster; statistically identical).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

COMPILE=1
FUSED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --figures) FIGURES="$2"; shift 2 ;;
    --seeds) SEEDS="${2//,/ }"; FIGURE_SEEDS=""; shift 2 ;;
    --full) FIGURE_SEEDS=""; shift ;;
    --gpus) read -r -a GPU_LIST <<< "${2//,/ }"; shift 2 ;;
    --no-compile) COMPILE=0; shift ;;
    --fused) FUSED=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
need_cache
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# every seed used must have both checkpoints
all_seeds="$SEEDS"
for item in $FIGURE_SEEDS; do
  all_seeds="$all_seeds ${item#*=}"
done
missing=0
for seed in $(echo "${all_seeds//,/ }" | tr ' ' '\n' | sort -nu); do
  for pattern in "$E0_PATTERN" "$E2_PATTERN"; do
    if [[ -z "$(compgen -G "${pattern//\{seed\}/$seed}" || true)" ]]; then
      echo "ERROR: no checkpoint for seed $seed: ${pattern//\{seed\}/$seed}" >&2
      missing=1
    fi
  done
done
(( missing )) && { echo "train them with 03_train.sh or set E0_PATTERN / E2_PATTERN" >&2; exit 1; }

read -r -a figure_list <<< "$FIGURES"
read -r -a seed_list <<< "$SEEDS"
read -r -a figure_seed_list <<< "$FIGURE_SEEDS"
args=(--cache-dir "$CACHE_DIR" --e0 "$E0_PATTERN" --e2 "$E2_PATTERN" --out-dir "$RESULTS_DIR"
      --figures "${figure_list[@]}" --seeds "${seed_list[@]}")
[[ ${#figure_seed_list[@]} -gt 0 ]] && args+=(--figure-seeds "${figure_seed_list[@]}")
[[ -n "${EVAL_BATCH:-}" ]] && args+=(--eval-batch "$EVAL_BATCH")   # images per batch; lower it on small GPUs
if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
  args+=(--gpus "${GPU_LIST[@]}")
  if (( FUSED )); then
    args+=(--fused)
  elif (( COMPILE )); then
    args+=(--compile)
  fi
else
  echo "WARNING: no GPU found; the missions run on the CPU (days, not hours)" >&2
  args+=(--gpus "")
fi

run_log="$LOG_DIR/missions_$(date +%Y%m%d_%H%M%S).log"
say "missions: figures $FIGURES; results in $RESULTS_DIR; progress in $run_log"
"$PY" -u "$SPACE_REPRODUCE" "${args[@]}" 2>&1 | tee "$run_log"
failed=$(grep -c FAILED "$run_log" || true)
n_done=$(find "$RESULTS_DIR" -name '*.done' | wc -l | tr -d ' ')
say "$n_done jobs finished in $RESULTS_DIR"
if [[ "$failed" -gt 0 ]]; then
  grep FAILED "$run_log" >&2
  echo "WARNING: $failed job(s) failed; see <config>_seed<k>.log in $RESULTS_DIR." >&2
  echo "Rerunning this script retries only the jobs without a .done file." >&2
  exit 1
fi
say "next: bash scripts/sdc_coins26/05_report.sh"
