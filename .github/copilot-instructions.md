# Copilot Instructions

## Commands

### Environment setup

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson-rtmpose.txt
```

### Run the current prototype

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_cameras.py --max-frames 120 --save-every 30
```

### Run with live display

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_cameras.py --display
```

### Build / test / lint status

This repository does not currently include a committed build system, automated test suite, or lint configuration, so there is no single-test command yet.

## High-level architecture

`fitra-cam` is currently a minimal Jetson-side prototype for dual-camera 2D pose estimation. The only runnable application in the repository today is `scripts/dual_rtmpose_cameras.py`.

That script:

- opens exactly two USB cameras, defaulting to the known `/dev/v4l/by-path/...` device names from the checked Jetson setup
- uses apt-provided OpenCV `CAP_GSTREAMER` with MJPEG camera pipelines for capture
- creates one `rtmlib.PoseTracker` per camera
- processes frames as `capture -> pose tracking/inference -> annotation -> optional display/save`
- saves annotated snapshots under `outputs/dual_rtmpose/`
- prints a per-camera runtime summary with frame counts, failures, and average FPS

The broader architecture is captured in `docs/README.md` and `docs/research/*.md`. Those documents describe the intended next stages of the system:

1. multi-camera ingest on Jetson
2. timestamp-based soft sync and camera calibration
3. RTMDet + RTMPose 2D inference with detector decimation and tracking between detections
4. confidence-weighted multi-view triangulation plus temporal smoothing for 3D pose
5. WebSocket delivery from the Jetson side and Three.js-based 3D visualization in a Web UI
6. stage-by-stage benchmarking using per-frame timestamps and `tegrastats`

## Key repository conventions

- Treat this as a Jetson-specific codebase. The checked environment and docs assume Jetson Orin Nano Super, JetPack 6.2.x, and Python 3.10.
- Prefer persistent camera identifiers under `/dev/v4l/by-path`. Do not switch to raw `/dev/video*` names or `/dev/v4l/by-id` without re-validating on hardware; the docs record that `by-path` was the stable choice on the target machine.
- The current prototype uses apt-provided `python3-opencv` so OpenCV `CAP_GSTREAMER` remains available. Prefer Jetson-friendly MJPEG pipelines (`nvjpegdec` by default, `nvv4l2decoder` as an option) and keep using `/dev/v4l/by-path` identifiers for stable camera selection.
- Runtime defaults are conservative bring-up settings: `--mode lightweight`, `--det-frequency 5`, `onnxruntime`, CPU-oriented execution, headless by default unless `--display` is passed.
- When extending toward the planned 3D pipeline, preserve timestamped frame flow and latest-frame-first behavior. The research docs consistently prefer real-time freshness over processing every queued frame.
- Use the research docs as the source of truth for architecture decisions that are not yet implemented in code. Those docs are written in Japanese and contain important project-specific hardware findings and roadmap decisions.
