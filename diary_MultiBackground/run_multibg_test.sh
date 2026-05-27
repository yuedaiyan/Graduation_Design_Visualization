#!/usr/bin/env bash
# Quick 100-item test run per background.
# Prints per-step timing and an estimated total time for the full run.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 diary_MultiBackground/main.py --test
