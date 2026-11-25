#!/usr/bin/env bash
# A few-minute check that the whole pipeline runs, before spending GPU-hours on it: train E0
# and E2 for one epoch on 300 images, fly two arms (the naive chip and the fully mitigated
# one) on 300 test images, then summarize and plot. The accuracies are not meaningful.
#
#   bash scripts/sdc_coins26/smoke_test.sh        # needs 01_download.sh and 02_build_cache.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_cache
need_file "$PRESTO_WEIGHTS" "Presto weights" "01_download.sh"
SMOKE="$SDC_WORK/smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
export CUDA_VISIBLE_DEVICES="${GPU_LIST[0]:-}"
run() { "$PY" -W ignore "$SPACE_MAIN" "$@"; }

say "1/4 E0 fine-tuning (1 epoch, 300 images)"
run finetune --cache-dir "$CACHE_DIR" --presto-weights "$PRESTO_WEIGHTS" --seed 0 \
  --limit 300 --max-epochs 1 --out "$SMOKE/e0_seed0.pt"
say "2/4 E2 noise-aware LoRA (1 epoch)"
run lora --cache-dir "$CACHE_DIR" --checkpoint "$SMOKE/e0_seed0.pt" --seed 0 \
  --limit 300 --max-epochs 1 --out "$SMOKE/e2_seed0.pt"
say "3/4 missions E1a (naive) and E4 (fully mitigated), 300 test images"
run mission --cache-dir "$CACHE_DIR" --arms E1a E4 --seeds 0 --test-limit 300 \
  --e0 "$SMOKE/e0_seed{seed}.pt" --e2 "$SMOKE/e2_seed{seed}.pt" --out "$SMOKE/results.jsonl"
say "4/4 summary and figures"
run summarize "$SMOKE/results.jsonl" --paper none
if "$PY" -c "import matplotlib" 2>/dev/null; then
  mkdir -p "$MPLCONFIGDIR"
  run plot "$SMOKE/results.jsonl" --out-dir "$SMOKE/figures"
fi
say "smoke test passed ($SMOKE)"
