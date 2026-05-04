#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements-jetson-rtmpose.txt"
ORT_VARIANT="${ORT_VARIANT:-cpu}"
ORT_GPU_WHEEL_URL="${ORT_GPU_WHEEL_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl}"

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup][warn] %s\n' "$*" >&2
}

die() {
  printf '[setup][error] %s\n' "$*" >&2
  exit 1
}

package_location() {
  python -m pip show "$1" 2>/dev/null | awk -F': ' '$1 == "Location" {print $2; exit}'
}

uninstall_if_in_venv() {
  local package_name="$1"
  local location

  location="$(package_location "$package_name" || true)"
  if [[ -n "$location" && "$location" == "$VENV_DIR"/lib/* ]]; then
    log "Removing $package_name from $VENV_DIR"
    python -m pip uninstall -y "$package_name" >/dev/null
  fi
}

log "Repository root: $ROOT_DIR"

if [[ "$(uname -m)" != "aarch64" ]]; then
  warn "This script is intended for Jetson aarch64 systems."
fi

if [[ -f /etc/nv_tegra_release ]]; then
  log "Jetson release: $(tr '\n' ' ' </etc/nv_tegra_release | sed 's/[[:space:]]\+/ /g')"
else
  warn "Jetson release file not found: /etc/nv_tegra_release"
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"Python 3.10 is required on JetPack 6.2.1, found {sys.version.split()[0]}"
    )
PY

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  die "Missing requirements file: $REQUIREMENTS_FILE"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

log "Upgrading pip"
python -m pip install --upgrade pip >/dev/null

uninstall_if_in_venv onnxruntime-gpu
uninstall_if_in_venv onnxruntime
uninstall_if_in_venv opencv-python
uninstall_if_in_venv opencv-contrib-python
uninstall_if_in_venv opencv-python-headless
uninstall_if_in_venv opencv-contrib-python-headless

log "Installing Python dependencies"
python -m pip install -r "$REQUIREMENTS_FILE"
python -m pip install --no-deps rtmlib==0.0.15

case "$ORT_VARIANT" in
  cpu)
    ;;
  gpu)
    log "Installing JetPack 6.2.1 compatible onnxruntime-gpu wheel"
    python -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
    PYTHONNOUSERSITE=1 python -m pip install --no-deps "$ORT_GPU_WHEEL_URL"
    ;;
  *)
    die "ORT_VARIANT must be either 'cpu' or 'gpu'"
    ;;
esac

USER_SITE_OPENCV="$(python - <<'PY'
import site
from pathlib import Path

user_site = Path(site.getusersitepackages())
for name in (
    "cv2",
    "opencv_python.libs",
    "opencv_contrib_python.libs",
):
    if (user_site / name).exists():
        print(user_site)
        break
PY
)"
if [[ -n "${USER_SITE_OPENCV:-}" ]]; then
  warn "OpenCV wheels are present in user site-packages: $USER_SITE_OPENCV"
  warn "Runtime scripts already force PYTHONNOUSERSITE=1, but keep that env var for manual checks too."
fi

log "Verifying runtime imports"
PYTHONNOUSERSITE=1 python - <<'PY'
import sys

import cv2
import onnxruntime as ort

print(f"[verify] python: {sys.executable}")
print(f"[verify] cv2: {cv2.__version__} ({cv2.__file__})")
print(f"[verify] onnxruntime: {ort.__version__} ({ort.__file__})")

gstreamer_line = next(
    (line.strip() for line in cv2.getBuildInformation().splitlines() if "GStreamer:" in line),
    "GStreamer: UNKNOWN",
)
print(f"[verify] {gstreamer_line}")
providers = ort.get_available_providers()
print(f"[verify] providers: {providers}")

if "YES" not in gstreamer_line:
    raise SystemExit(
        "OpenCV was imported without GStreamer support. Ensure apt's python3-opencv is installed and rerun with PYTHONNOUSERSITE=1."
    )

if "CUDAExecutionProvider" not in providers:
    print(
        "[verify] note: CUDAExecutionProvider is unavailable in the active environment."
    )
    print(
        "[verify] note: For JetPack 6.2.1 GPU execution, use a cu126/cuDNN9-compatible "
        "onnxruntime-gpu wheel such as Jetson AI Lab's 1.23.0 build."
    )
PY

log "Environment is ready."
if [[ "$ORT_VARIANT" == "gpu" ]]; then
  log "Run the prototype with:"
  log "  . .venv/bin/activate && python scripts/dual_rtmpose_cameras.py --device auto --max-frames 120 --save-every 30"
  log "Run the Web UI with:"
  log "  . .venv/bin/activate && python scripts/dual_rtmpose_web.py --device auto --host 0.0.0.0 --port 8000"
else
  log "Run the prototype with:"
  log "  . .venv/bin/activate && python scripts/dual_rtmpose_cameras.py --device cpu --max-frames 120 --save-every 30"
  log "Run the Web UI with:"
  log "  . .venv/bin/activate && python scripts/dual_rtmpose_web.py --device cpu --host 0.0.0.0 --port 8000"
fi
