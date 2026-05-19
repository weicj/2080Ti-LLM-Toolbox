#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname
uname -a

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,compute_cap,driver_version,memory.total --format=csv
  nvidia-smi topo -m || true
else
  echo "nvidia-smi not found"
fi

echo
echo "== python =="
if command -v python >/dev/null 2>&1; then
  python --version
  python - <<'PY' || true
import importlib.metadata as md
for name in ["torch", "vllm", "sglang", "flashinfer-python", "flashinfer", "triton"]:
    try:
        print(f"{name}: {md.version(name)}")
    except Exception:
        pass
PY
else
  echo "python not found"
fi

echo
echo "== cuda =="
command -v nvcc >/dev/null 2>&1 && nvcc --version || echo "nvcc not found"

