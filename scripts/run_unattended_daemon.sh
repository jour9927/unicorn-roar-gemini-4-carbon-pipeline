#!/usr/bin/env bash
# Unattended Production Launcher for 獨角獸那麼叫 (@unicorn-roar)
# Automatically syncs latest master scripts on local host, executes full Long + Short episode
# generation (offloading heavy GPU tasks to RTX PRO 6000 / b850 when online), and logs output.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/unattended_${TS}.log"

echo "[${TS}] Starting unattended @unicorn-roar Long + Short production..." | tee -a "${LOG_FILE}"
"${REPO_ROOT}/.venv/bin/unicorn-roar" run "$@" 2>&1 | tee -a "${LOG_FILE}"
echo "[$(date +%Y%m%d_%H%M%S)] Production finished successfully." | tee -a "${LOG_FILE}"
