#!/usr/bin/env bash
set -euo pipefail

echo "== nvidia-smi =="
nvidia-smi

echo
echo "== topology =="
nvidia-smi topo -m || true

echo
echo "== clocks/power/temp =="
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,temperature.gpu,power.draw,power.limit,memory.used,memory.total,clocks.sm,clocks.mem --format=csv,noheader,nounits

echo
echo "== driver =="
cat /proc/driver/nvidia/version 2>/dev/null || true

