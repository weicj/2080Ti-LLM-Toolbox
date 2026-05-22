#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/vllm_server.log [awq|gptq]" >&2
  exit 2
fi

LOG_PATH="$1"
MODE="${2:-awq}"

if [[ ! -f "$LOG_PATH" ]]; then
  echo "log file not found: $LOG_PATH" >&2
  exit 2
fi

mode_lc="$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
if [[ "$mode_lc" != "awq" && "$mode_lc" != "gptq" ]]; then
  echo "invalid mode: $MODE (expected awq or gptq)" >&2
  exit 2
fi

has_marlin_kernel=0
has_awq_marlin=0
has_gptq_marlin=0
has_slow_awq=0
has_slow_gptq=0

if rg -qi 'MarlinLinearKernel|marlin' "$LOG_PATH"; then
  has_marlin_kernel=1
fi
if rg -qi 'awq_marlin' "$LOG_PATH"; then
  has_awq_marlin=1
fi
if rg -qi 'gptq_marlin' "$LOG_PATH"; then
  has_gptq_marlin=1
fi

# Heuristic slow-path indicators.
if rg -qi 'quantization[^\\n]*awq' "$LOG_PATH" && ! rg -qi 'awq_marlin' "$LOG_PATH"; then
  has_slow_awq=1
fi
if rg -qi 'quantization[^\\n]*gptq' "$LOG_PATH" && ! rg -qi 'gptq_marlin' "$LOG_PATH"; then
  has_slow_gptq=1
fi

echo "log=$LOG_PATH"
echo "mode=$mode_lc"
echo "has_marlin_kernel=$has_marlin_kernel"
echo "has_awq_marlin=$has_awq_marlin"
echo "has_gptq_marlin=$has_gptq_marlin"
echo "has_slow_awq=$has_slow_awq"
echo "has_slow_gptq=$has_slow_gptq"

if [[ "$mode_lc" == "awq" ]]; then
  if [[ "$has_marlin_kernel" -eq 1 && "$has_awq_marlin" -eq 1 && "$has_slow_awq" -eq 0 ]]; then
    echo "decision=PASS"
    exit 0
  fi
else
  if [[ "$has_marlin_kernel" -eq 1 && "$has_gptq_marlin" -eq 1 && "$has_slow_gptq" -eq 0 ]]; then
    echo "decision=PASS"
    exit 0
  fi
fi

echo "decision=FAIL"
exit 1
