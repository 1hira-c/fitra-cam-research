#!/usr/bin/env python3
"""Record both cameras for --seconds, then overlay RTMPose skeleton on the
recorded files. Useful for post-hoc keypoint quality review without
needing the pose pipeline to keep up with capture in real time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # noqa: E402
    CAM_COLORS,
    CameraConfig,
    PoseDrawer,
    add_common_args,
    build_engines_for,
    open_v4l2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record dual cameras and overlay RTMPose")
    add_common_args(parser)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", default="outputs/recorded_rtmpose")
    return parser.parse_args()


def _record_one(cfg: CameraConfig, out_path: Path, duration_s: float, stop: threading.Event) -> tuple[int, float]:
    """Record raw frames to out_path. Returns (frame_count, actual_fps).

    Because two USB 2.0 cameras typically share bandwidth and deliver
    less than the requested fps, we buffer frames during capture and
    write with measured fps so playback timing matches reality.
    """
    cap = open_v4l2(cfg)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    buf: list[np.ndarray] = []
    timestamps: list[float] = []
    start = time.monotonic()
    try:
        while not stop.is_set() and (time.monotonic() - start) < duration_s:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            buf.append(frame)
            timestamps.append(time.monotonic())
    finally:
        cap.release()
    n = len(buf)
    if n < 2:
        raise RuntimeError(f"recording too short from {cfg.path}: got {n} frames")
    actual_fps = (n - 1) / (timestamps[-1] - timestamps[0])
    writer = cv2.VideoWriter(str(out_path), fourcc, float(actual_fps), (cfg.width, cfg.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter at {out_path}")
    try:
        for f in buf:
            writer.write(f)
    finally:
        writer.release()
    return n, actual_fps


def _overlay_one(raw_path: Path, out_path: Path, engine, drawer: PoseDrawer, color) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {raw_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open VideoWriter at {out_path}")
    seq = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            seq += 1
            result = engine.process(seq, frame, time.monotonic())
            drawer.draw(frame, result.persons, color)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return seq, float(fps), w, h


def _side_by_side(overlay_paths: list[Path], out_path: Path) -> int:
    caps = [cv2.VideoCapture(str(p)) for p in overlay_paths]
    try:
        if not all(c.isOpened() for c in caps):
            raise RuntimeError("failed to reopen overlay clips for side-by-side composition")
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
        w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter at {out_path}")
        n = 0
        try:
            while True:
                frames = []
                ok_all = True
                for c in caps:
                    ok, f = c.read()
                    if not ok or f is None:
                        ok_all = False
                        break
                    if f.shape[1] != w or f.shape[0] != h:
                        f = cv2.resize(f, (w, h))
                    frames.append(f)
                if not ok_all:
                    break
                writer.write(np.hstack(frames))
                n += 1
        finally:
            writer.release()
        return n
    finally:
        for c in caps:
            c.release()


def main() -> int:
    args = parse_args()
    cam_cfgs = [
        CameraConfig(path=args.cam0, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc),
        CameraConfig(path=args.cam1, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc),
    ]

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = [out_dir / f"raw_cam{i}.mp4" for i in range(2)]
    overlay_paths = [out_dir / f"overlay_cam{i}.mp4" for i in range(2)]
    side_path = out_dir / "overlay_side_by_side.mp4"

    print(f"[record] writing raw to {out_dir} for {args.seconds:.1f}s", file=sys.stderr)
    stop = threading.Event()
    results: dict[int, tuple[int, float]] = {}
    threads = []
    for i, (cfg, path) in enumerate(zip(cam_cfgs, raw_paths)):
        def runner(idx=i, cfg=cfg, path=path):
            results[idx] = _record_one(cfg, path, args.seconds, stop)
        t = threading.Thread(target=runner, name=f"Recorder-{i}", daemon=True)
        t.start()
        threads.append(t)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        print("[record] interrupted", file=sys.stderr)
        return 1
    for i, (n, fps) in sorted(results.items()):
        print(f"[record] cam{i}: {n} frames @ {fps:.2f}fps -> {raw_paths[i]}", file=sys.stderr)

    print(f"[overlay] building engines (device={args.device})...", file=sys.stderr)
    engines = build_engines_for(2, args)
    drawer = PoseDrawer(kp_thr=args.kp_thr)
    for i, (raw_p, out_p, engine) in enumerate(zip(raw_paths, overlay_paths, engines)):
        n, fps, w, h = _overlay_one(raw_p, out_p, engine, drawer, CAM_COLORS[i % len(CAM_COLORS)])
        print(f"[overlay] cam{i}: {n} frames @ {fps:.1f}fps {w}x{h} -> {out_p}", file=sys.stderr)

    print(f"[merge] writing {side_path}", file=sys.stderr)
    n_side = _side_by_side(overlay_paths, side_path)
    print(f"[merge] wrote {n_side} frames -> {side_path}", file=sys.stderr)
    print(f"[done] outputs at {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
