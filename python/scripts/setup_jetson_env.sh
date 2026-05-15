#!/usr/bin/env bash
# Minimal Jetson env setup. Assumes:
#   - apt python3-opencv is installed (provides cv2.CAP_V4L2)
#   - .venv exists and was created with --system-site-packages
#   - onnxruntime-gpu is either already present, or will be installed
#     manually from Jetson AI Lab when the user wants GPU execution
#
# Run from anywhere:
#   chmod +x python/scripts/setup_jetson_env.sh
#   ./python/scripts/setup_jetson_env.sh
#
# The .venv is created at python/.venv (alongside requirements-jetson.txt),
# not at the top of the repo, because the C++ migration owns the repo root.

set -euo pipefail

PY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PY_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "[setup] creating .venv with --system-site-packages"
  python3 -m venv --system-site-packages .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt

echo "[setup] verifying runtime..."
python - <<'PY'
import cv2
import numpy as np
import onnxruntime as ort

print(f"cv2:    {cv2.__version__}  ({cv2.__file__})")
print(f"numpy:  {np.__version__}")
print(f"ort:    {ort.__version__}")
print(f"providers: {ort.get_available_providers()}")
PY

echo "[setup] done."
echo
echo "If you need CUDA / TensorRT execution providers, install the Jetson AI Lab"
echo "wheel after this script (see README)."
