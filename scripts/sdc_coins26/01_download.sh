#!/usr/bin/env bash
# Step 1: download the inputs (about 2 GB; everything already present is kept).
#
#   bash scripts/sdc_coins26/01_download.sh
#   bash scripts/sdc_coins26/01_download.sh --force    # download and extract again
#
#  * EuroSAT multispectral (Sentinel-2, 13 bands, 27,000 images): torchgeo's Hugging Face
#    mirror, or Zenodo record 7711810 if that fails. Set EUROSAT_ZIP to use a zip you already have.
#  * The official EuroSAT train / val / test split files (16,200 / 5,400 / 5,400 images).
#  * Presto's pretrained weights (default_model.pt, 0.4 M parameters) from nasaharvest/presto.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

FORCE=""
[[ "${1:-}" == "--force" ]] && FORCE=1
mkdir -p "$DATA_DIR" "$SPLIT_DIR" "$EUROSAT_DIR"

fetch() {  # fetch OUTPUT URL [URL...]: the first URL that works
  local out="$1"; shift
  local progress=(-sS)
  [[ -t 2 ]] && progress=(--progress-bar)
  for url in "$@"; do
    if curl -fL --retry 3 -C - "${progress[@]}" -o "$out" "$url"; then
      return 0
    fi
    echo "  could not fetch $url" >&2
  done
  return 1
}

for split in train val test; do
  f="$SPLIT_DIR/eurosat-$split.txt"
  if [[ -s "$f" && -z "$FORCE" ]]; then
    say "[splits] $f present ($(wc -l < "$f") images)"
  else
    say "[splits] eurosat-$split.txt"
    fetch "$f" "https://hf.co/datasets/torchgeo/eurosat/resolve/main/eurosat-$split.txt"
  fi
done

if [[ -s "$PRESTO_WEIGHTS" && -z "$FORCE" ]]; then
  say "[presto] $PRESTO_WEIGHTS present"
else
  say "[presto] default_model.pt"
  fetch "$PRESTO_WEIGHTS" "https://raw.githubusercontent.com/nasaharvest/presto/main/data/default_model.pt"
fi

n_tif=$(find "$EUROSAT_DIR" -name '*.tif' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n_tif" -ge 27000 && -z "$FORCE" ]]; then
  say "[eurosat] $n_tif images already in $EUROSAT_DIR"
else
  if [[ ! -s "$EUROSAT_ZIP" || -n "$FORCE" ]]; then
    say "[eurosat] downloading EuroSAT_MS (about 2 GB)"
    fetch "$EUROSAT_ZIP" \
      "https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip" \
      "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
  fi
  say "[eurosat] extracting $EUROSAT_ZIP"
  unzip -q -o "$EUROSAT_ZIP" -d "$EUROSAT_DIR"
  n_tif=$(find "$EUROSAT_DIR" -name '*.tif' | wc -l | tr -d ' ')
  if [[ "$n_tif" -lt 27000 ]]; then
    echo "ERROR: expected 27,000 GeoTIFFs in $EUROSAT_DIR, found $n_tif" >&2
    exit 1
  fi
  say "[eurosat] $n_tif images extracted"
fi
say "inputs ready; next: bash scripts/sdc_coins26/02_build_cache.sh"
