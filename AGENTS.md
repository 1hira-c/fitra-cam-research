# Repository Guidelines

## Project Structure & Module Organization

`fitra-cam` is a Jetson-side prototype for dual USB-camera RTMPose inference. Core Python runtime code lives in `scripts/`: `dual_rtmpose_core.py` contains shared capture, inference, publishing, and configuration logic; `dual_rtmpose_cameras.py` is the CLI snapshot/display runner; `dual_rtmpose_web.py` serves the skeleton-only WebSocket UI. The static web client is in `web/dual_rtmpose/`. Research notes and hardware decisions are in `docs/README.md` and `docs/research/*.md`. Generated snapshots go under `outputs/dual_rtmpose/` and should not be treated as source.

## Build, Test, and Development Commands

Set up the Jetson environment:

```bash
chmod +x scripts/setup_jetson_env.sh
./scripts/setup_jetson_env.sh
```

Run a short dual-camera smoke test:

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_cameras.py --device auto --max-frames 120 --save-every 30
```

Run with display windows:

```bash
python scripts/dual_rtmpose_cameras.py --device auto --display
```

Run the web UI:

```bash
python scripts/dual_rtmpose_web.py --device auto --host 0.0.0.0 --port 8000
```

There is currently no committed build system, automated test suite, or lint command.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code with 4-space indentation, type hints where they clarify interfaces, and `dataclass` objects for structured runtime configuration. Keep CLI flags lowercase and hyphenated, matching existing options such as `--det-frequency` and `--save-every`. Prefer stable `/dev/v4l/by-path/...` camera identifiers over raw `/dev/video*` names unless hardware has been revalidated. Keep the runtime compatible with apt-provided OpenCV so `CAP_GSTREAMER` remains available; do not add pip OpenCV wheels to the venv.

## Testing Guidelines

Until automated tests are added, verify changes with a short `--max-frames` run on the Jetson target. For web changes, start `dual_rtmpose_web.py` and confirm `http://JETSON_IP:8000/` receives pose bundles and renders both camera panes. For environment or backend changes, check OpenCV GStreamer support and ONNX Runtime providers before benchmarking.

## Commit & Pull Request Guidelines

Git history is not available in this workspace, so use clear imperative commit messages such as `Add WebSocket pose metrics` or `Fix GStreamer decoder fallback`. Pull requests should describe the hardware tested, commands run, observed FPS or latency impact when relevant, and any changes to camera paths, JetPack assumptions, dependencies, or generated outputs. Include screenshots for visible web UI changes.

## Security & Configuration Tips

Do not commit local camera snapshots, large benchmark outputs, or machine-specific secrets. Keep Jetson-specific dependency notes in `README.md` or `docs/research/` when setup assumptions change.
