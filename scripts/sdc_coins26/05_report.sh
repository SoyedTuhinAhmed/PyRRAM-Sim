#!/usr/bin/env bash
# Step 5: compare the results with the paper and draw the figures.
#
#   bash scripts/sdc_coins26/05_report.sh
#
# Writes
#   $RESULTS_DIR/summary.txt   every point: n, mean, std, the paper's 10-seed mean / std, and z
#   $RESULTS_DIR/summary.csv   the same table for a spreadsheet
#   $FIGURES_DIR/*.png         accuracy vs mission time (Figs. 2, 5), dose (6, 7), temperature (8)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

shopt -s nullglob
results=("$RESULTS_DIR"/*.jsonl)
if [[ ${#results[@]} -eq 0 ]]; then
  echo "ERROR: no results in $RESULTS_DIR (run 04_missions.sh first)" >&2
  exit 1
fi
mkdir -p "$FIGURES_DIR" "$MPLCONFIGDIR"

"$PY" -W ignore "$SPACE_MAIN" summarize "${results[@]}" --csv "$RESULTS_DIR/summary.csv" \
  > "$RESULTS_DIR/summary.txt"

echo "Headline points (12 months; mean over seeds, then the paper's mean / std and z):"
"$PY" -W ignore "$SPACE_MAIN" summarize "${results[@]}" --worst 0 \
  --arms E0 E2 E1a E1b E3_noLoRA E3 E4_head E4 E4_noLoRA P4Y RP4Y R4Y S4Y \
  | awk 'NR == 1 || $2 == "12mo" || $2 == "digital"'
echo
sed -n '/points compared with the paper/,$p' "$RESULTS_DIR/summary.txt" | grep -v "^wrote"
echo
if "$PY" -c "import matplotlib" 2>/dev/null; then
  "$PY" -W ignore "$SPACE_MAIN" plot "${results[@]}" --out-dir "$FIGURES_DIR"
else
  echo "matplotlib is not installed: no figures (pip install matplotlib)"
fi
say "full table: $RESULTS_DIR/summary.txt ($RESULTS_DIR/summary.csv)"
