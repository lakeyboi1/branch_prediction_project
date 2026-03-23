#!/usr/bin/env bash
set -euo pipefail

SWEEP_CMDS="${1:-sweep_commands.txt}"
PLOT_CMDS="${2:-plot_commands.txt}"

if [[ ! -f "${SWEEP_CMDS}" ]]; then
  echo "Missing ${SWEEP_CMDS}"
  exit 1
fi

if [[ ! -f "${PLOT_CMDS}" ]]; then
  echo "Missing ${PLOT_CMDS}"
  exit 1
fi

run_cmd_file () {
  local f="$1"
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    echo ""
    echo ">>> ${line}"
    bash -lc "${line}"
  done < "${f}"
}

run_cmd_file "${SWEEP_CMDS}"
run_cmd_file "${PLOT_CMDS}"