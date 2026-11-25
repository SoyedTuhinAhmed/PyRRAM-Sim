#!/usr/bin/env bash
# The whole reproduction, from an empty folder to the comparison with the paper:
# download -> cache -> smoke test -> train 10 seeds -> missions -> report.
# Every step keeps what is already done, so rerun the script after an interruption.
#
#   bash scripts/sdc_coins26/run_all.sh
#   bash scripts/sdc_coins26/run_all.sh --full      # 10 seeds for every figure
#
# On 8 RTX A5500 GPUs: about 1.5 h of training and 2 h 10 min of missions
# (5.5 h with --full). Settings: scripts/sdc_coins26/config.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$HERE/01_download.sh"
bash "$HERE/02_build_cache.sh"
bash "$HERE/smoke_test.sh"
bash "$HERE/03_train.sh"
bash "$HERE/04_missions.sh" "$@"
bash "$HERE/05_report.sh"
