#!/usr/bin/env bash
# Step 3: the two design-time models of every seed, one seed per GPU at a time.
#
#   E0  Presto fine-tuned on EuroSAT in FP32 (AdamW 3e-4, early stopping, ~45 min on one GPU)
#   E2  E0 + noise-aware LoRA (rank 8, alpha 16, weight noise 0.1 max|W|), merged into the
#       weights so the analog chip holds one weight matrix per layer (~15 min)
#
#   bash scripts/sdc_coins26/03_train.sh                     # seeds 0-9 on every GPU
#   bash scripts/sdc_coins26/03_train.sh --seeds 0,1 --gpus 0,1
#
# Finished checkpoints are kept, so the script can be rerun after an interruption. To use
# existing checkpoints instead (for example the ones of the original case study), skip this
# step and set E0_PATTERN / E2_PATTERN (see config.sh). TRAIN_ARGS is passed to both commands,
# e.g. TRAIN_ARGS="--limit 300 --max-epochs 1" for a quick check.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="${2//,/ }"; shift 2 ;;
    --gpus) read -r -a GPU_LIST <<< "${2//,/ }"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
read -r -a SEED_LIST <<< "$SEEDS"
need_cache
need_file "$PRESTO_WEIGHTS" "Presto weights" "01_download.sh"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

target() {  # target PATTERN SEED: the checkpoint path of a seed (an existing match, or the path to write)
  local path="${1//\{seed\}/$2}"
  local hit
  hit="$(compgen -G "$path" | sort | tail -n 1 || true)"
  if [[ -n "$hit" ]]; then
    echo "$hit"
  elif [[ "$path" == *[\*\?\[]* ]]; then
    echo "ERROR: no checkpoint matches $path, and a glob cannot be a training target" >&2
    return 1
  else
    echo "$path"
  fi
}

train_seed() {  # train_seed SEED GPU
  local seed="$1" gpu="$2" e0 e2 log="$LOG_DIR/train_seed$1.log"
  e0="$(target "$E0_PATTERN" "$seed")"
  e2="$(target "$E2_PATTERN" "$seed")"
  if [[ ! -s "$e0" ]]; then
    say "seed $seed: E0 fine-tuning on GPU ${gpu:-cpu} (log $log)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -W ignore "$SPACE_MAIN" finetune --cache-dir "$CACHE_DIR" \
      --presto-weights "$PRESTO_WEIGHTS" --seed "$seed" --out "$e0.tmp" ${TRAIN_ARGS:-} >> "$log" 2>&1
    mv "$e0.tmp" "$e0"
  fi
  if [[ ! -s "$e2" ]]; then
    say "seed $seed: E2 noise-aware LoRA on GPU ${gpu:-cpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -W ignore "$SPACE_MAIN" lora --cache-dir "$CACHE_DIR" \
      --checkpoint "$e0" --seed "$seed" --out "$e2.tmp" ${TRAIN_ARGS:-} >> "$log" 2>&1
    mv "$e2.tmp" "$e2"
  fi
  say "seed $seed: done ($e0, $e2)"
}

workers=("${GPU_LIST[@]:-}")
[[ ${#GPU_LIST[@]} -eq 0 ]] && { echo "WARNING: no GPU found; training on the CPU is very slow" >&2; workers=(""); }
say "${#SEED_LIST[@]} seeds on ${#workers[@]} worker(s): GPU ${workers[*]:-cpu}"
pids=()
for w in "${!workers[@]}"; do
  (
    set -e
    for i in "${!SEED_LIST[@]}"; do
      if (( i % ${#workers[@]} == w )); then
        train_seed "${SEED_LIST[$i]}" "${workers[$w]}"
      fi
    done
  ) &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done

echo
echo "test accuracy (paper, 10 seeds: E0 0.957 +- 0.004, E2 0.905 +- 0.013):"
for seed in "${SEED_LIST[@]}"; do
  log="$LOG_DIR/train_seed$seed.log"
  # the last E0 and E2 results (a rerun appends to the log)
  [[ -f "$log" ]] && grep "^test:" "$log" | tail -n 2 | sed "s/^/  seed $seed /" || true
done
if (( failed )); then
  echo "ERROR: $failed worker(s) failed; see $LOG_DIR/train_seed*.log and rerun to continue" >&2
  exit 1
fi
say "checkpoints ready; next: bash scripts/sdc_coins26/04_missions.sh"
