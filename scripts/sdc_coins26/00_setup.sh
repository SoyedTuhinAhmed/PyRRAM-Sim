#!/usr/bin/env bash
# Step 0: a Python environment with what the space case study needs.
#
#   bash scripts/sdc_coins26/00_setup.sh            # creates .venv in the repository
#   PY=/path/to/python bash .../00_setup.sh --no-venv  # installs into an existing environment
#
# Installs PyTorch (the default wheel of your platform; for a specific CUDA build see
# https://pytorch.org/get-started/locally and install torch first), einops (Presto),
# rasterio + pyproj (reading the EuroSAT GeoTIFFs once) and matplotlib (figures).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [[ "${1:-}" != "--no-venv" ]]; then
  VENV="$REPO_ROOT/.venv"
  if [[ ! -x "$VENV/bin/python" ]]; then
    say "creating $VENV"
    "$(command -v python3 || command -v python)" -m venv "$VENV"
  fi
  PY="$VENV/bin/python"
fi

say "installing into $("$PY" -c 'import sys; print(sys.prefix)')"
"$PY" -m pip install --upgrade pip
"$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install torch
"$PY" -m pip install numpy einops rasterio pyproj matplotlib

"$PY" - <<'EOF'
import torch, einops, rasterio, pyproj, matplotlib
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.device_count()} GPUs)" if torch.cuda.is_available() else ""))
EOF
say "environment ready; next: bash scripts/sdc_coins26/01_download.sh"
