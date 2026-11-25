#!/usr/bin/env bash
# Step 2: read the 27,000 GeoTIFFs once into an array cache (bands, coordinates, labels; 2.7 GB).
# Takes about a minute with 16 worker processes; every later step reads only the cache.
#
#   bash scripts/sdc_coins26/02_build_cache.sh
#   WORKERS=32 bash scripts/sdc_coins26/02_build_cache.sh --force
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [[ -s "$CACHE_DIR/manifest.json" && "${1:-}" != "--force" ]]; then
  say "cache present at $CACHE_DIR (--force to rebuild)"
  exit 0
fi
for split in train val test; do
  need_file "$SPLIT_DIR/eurosat-$split.txt" "split file" "01_download.sh"
done
say "building the cache in $CACHE_DIR"
"$PY" -W ignore "$SPACE_MAIN" build-cache --eurosat-root "$EUROSAT_DIR" --split-dir "$SPLIT_DIR" \
  --cache-dir "$CACHE_DIR" --workers "${WORKERS:-16}"
say "cache ready; next: bash scripts/sdc_coins26/03_train.sh"
