# shellcheck shell=bash disable=SC2034
# Shared settings for the SDC (COINS'26) reproduction scripts. Sourced by the other scripts,
# never run on its own. Every setting can be overridden from the environment, for example
#
#   SDC_WORK=/scratch/sdc GPUS="0 1" bash scripts/sdc_coins26/04_missions.sh

SDC_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SDC_SCRIPTS/../.." && pwd)"

# --- folders ---------------------------------------------------------------
SDC_WORK="${SDC_WORK:-$REPO_ROOT/sdc_work}"           # everything the scripts write (~8 GB)
DATA_DIR="${DATA_DIR:-$SDC_WORK/data}"                # EuroSAT zip + tifs, split files, Presto weights
CACHE_DIR="${CACHE_DIR:-$SDC_WORK/cache}"             # EuroSAT array cache (2.7 GB)
CKPT_DIR="${CKPT_DIR:-$SDC_WORK/checkpoints}"         # e0_seed<k>.pt, e2_seed<k>.pt
RESULTS_DIR="${RESULTS_DIR:-$SDC_WORK/results}"       # <arm>_seed<k>.jsonl / .log / .done
FIGURES_DIR="${FIGURES_DIR:-$SDC_WORK/figures}"
LOG_DIR="${LOG_DIR:-$SDC_WORK/logs}"

PRESTO_WEIGHTS="${PRESTO_WEIGHTS:-$DATA_DIR/default_model.pt}"
SPLIT_DIR="${SPLIT_DIR:-$DATA_DIR/splits}"
EUROSAT_DIR="${EUROSAT_DIR:-$DATA_DIR/eurosat_ms}"
EUROSAT_ZIP="${EUROSAT_ZIP:-$DATA_DIR/EuroSATallBands.zip}"

# Checkpoints of the E0 (FP32 fine-tuned) and E2 (noise-aware LoRA, merged) models, {seed} is
# substituted and globs are allowed. Point these at existing checkpoints to skip training, e.g.
#   E0_PATTERN='/path/results/e0_finetune/*_seed{seed}/checkpoints/best.pt'
# (the defaults are built first: a "}" inside ${VAR:-...} would end the expansion)
default_e0="$CKPT_DIR/e0_seed{seed}.pt"
default_e2="$CKPT_DIR/e2_seed{seed}.pt"
E0_PATTERN="${E0_PATTERN:-$default_e0}"
E2_PATTERN="${E2_PATTERN:-$default_e2}"
for pattern in "$E0_PATTERN" "$E2_PATTERN"; do
  if [[ "$pattern" != *"{seed}"* ]]; then
    echo "ERROR: checkpoint pattern $pattern has no {seed}: every seed would use the same model" >&2
    exit 1
  fi
done

# --- seeds -----------------------------------------------------------------
# The paper uses 10 seeds (= 10 trained models x 10 simulated chips). FIGURE_SEEDS lowers the
# count for the expensive figures; the defaults are the ones of the reproduction in the README.
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
FIGURE_SEEDS="${FIGURE_SEEDS:-fig5=0,1,2,3,4 fig6=0,1 fig7=0,1 fig8=0,1}"
FIGURES="${FIGURES:-fig2 fig3 fig5 fig6 fig7 fig8}"

# --- interpreter -----------------------------------------------------------
if [[ -z "${PY:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
  else
    PY="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$PY" ]]; then
  echo "ERROR: no python found; run scripts/sdc_coins26/00_setup.sh or set PY=/path/to/python" >&2
  exit 1
fi

# --- GPUs ------------------------------------------------------------------
# GPUS="0 1 2 3" picks devices; otherwise CUDA_VISIBLE_DEVICES, otherwise every GPU nvidia-smi lists.
if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
  fi
fi
read -r -a GPU_LIST <<< "${GPUS:-}"

# torch.compile writes kernels to these folders; they must allow executables (/tmp often does not).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SDC_WORK/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SDC_WORK/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SDC_WORK/.cache/matplotlib}"
export PYTHONUNBUFFERED=1

SPACE_MAIN="$REPO_ROOT/space_main.py"
SPACE_REPRODUCE="$REPO_ROOT/space_reproduce.py"

say() { echo "[$(date +%H:%M:%S)] $*"; }

need_file() {  # need_file PATH "what it is" "script that makes it"
  if [[ ! -s "$1" ]]; then
    echo "ERROR: $2 not found at $1 (run $3 first)" >&2
    exit 1
  fi
}

need_cache() {
  need_file "$CACHE_DIR/manifest.json" "EuroSAT cache" "02_build_cache.sh"
}
