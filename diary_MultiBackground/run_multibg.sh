#!/usr/bin/env bash
# Full MultiBackground run — 10,000 items per background.
# Interrupted runs resume automatically from cached vector shards.
# Output: output_All/diary_MultiBackground/try_N/
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="run_multibg_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
python3 diary_MultiBackground/main.py 2>&1 | tee "$LOG"
