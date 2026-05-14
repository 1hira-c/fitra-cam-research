#!/usr/bin/env python3
"""CLI: run RTMPose on two USB cameras, save annotated snapshots and/or display."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import threading
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # noqa: E402
    CAM_COLORS,
    CameraConfig,
    CameraReader,
    EngineStats,
    PoseDrawer,
    add_common_args,
    build_engines_for,
    update_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual USB cam RTMPose runner")
    add_common_args(parser)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="total processed frames across both cameras, 0 = infinite")
    parser.add_argument("--save-every", type=int, default=0,
                        help="save annotated jpg every N processed frames per camera, 0 = disabled")
    parser.add_argument("--display", action="store_true",
                        help="show OpenCV imshow windows (needs DISPLAY)")
    parser.add_argument("--output-dir", default="outputs/dual_rtmpose")
    return parser.parse_args()


def _worker_loop(
    cam_id: int,
    reader: CameraReader,
    engine,
    drawer: PoseDrawer,
    stats: EngineStats,
    last_frames: dict,
    shared: dict,
    args: argparse.Namespace,
    save_dir: Path,
):
    last_seq = 0
    processed_local = 0
    color = CAM_COLORS[cam_id % len(CAM_COLORS)]
    while not shared["stop"]:
        latest = reader.latest()
        if latest is None or latest[0] == last_seq:
            time.sleep(0.002)
            continue
        seq, frame, captured_at = latest
        last_seq = seq
        # working copy because we draw onto it
        frame_work = frame.copy()
        result = engine.process(seq, frame_work, captured_at)
        update_stats(stats, result)
        drawer.draw(frame_work, result.persons, color)
        last_frames[cam_id] = frame_work
        processed_local += 1
        with shared["lock"]:
            shared["processed_total"] += 1
            done = shared["processed_total"]
        if args.save_every and (processed_local % args.save_every == 0):
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"cam{cam_id}_{processed_local:06d}.jpg"
            cv2.imwrite(str(out_path), frame_work)
        if args.max_frames and done >= args.max_frames:
            shared["stop"] = True


def main() -> int:
    args = parse_args()

    cam_cfgs = [
        CameraConfig(path=args.cam0, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc),
        CameraConfig(path=args.cam1, width=args.width, height=args.height,
                     fps=args.fps, fourcc=args.fourcc),
    ]
    print(f"[setup] opening cameras: {args.cam0} / {args.cam1}", file=sys.stderr)
    readers = [CameraReader(i, c).start() for i, c in enumerate(cam_cfgs)]

    print(f"[setup] device={args.device}, building engines (this loads ONNX models)...", file=sys.stderr)
    engines = build_engines_for(len(readers), args)
    drawer = PoseDrawer(kp_thr=args.kp_thr)
    stats = [EngineStats() for _ in readers]

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / run_ts

    shared = {"stop": False, "processed_total": 0, "lock": threading.Lock()}
    last_frames: dict = {}

    workers = []
    for cam_id, (reader, engine) in enumerate(zip(readers, engines)):
        t = threading.Thread(
            target=_worker_loop,
            args=(cam_id, reader, engine, drawer, stats[cam_id],
                  last_frames, shared, args, save_dir),
            name=f"PoseWorker-{cam_id}",
            daemon=True,
        )
        t.start()
        workers.append(t)

    print(f"[setup] running. max_frames={args.max_frames}, save_every={args.save_every}, "
          f"save_dir={save_dir if args.save_every else '<disabled>'}", file=sys.stderr)

    last_log = time.monotonic()
    try:
        while not shared["stop"]:
            now = time.monotonic()
            if now - last_log >= args.log_every:
                for i, s in enumerate(stats):
                    pending = readers[i].seq - (s.pose_count)
                    print(
                        f"[stats] cam{i}: recv={readers[i].recv_fps:5.2f} "
                        f"avg_pose={s.avg_pose_fps:5.2f} recent_pose={s.recent_pose_fps:5.2f} "
                        f"stage_ms={s.last_stage_ms:6.1f} processed={s.pose_count} pending={max(pending, 0)}",
                        file=sys.stderr,
                    )
                last_log = now
            if args.display:
                for cam_id, frame in list(last_frames.items()):
                    cv2.imshow(f"cam{cam_id}", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    shared["stop"] = True
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        shared["stop"] = True

    for w in workers:
        w.join(timeout=2.0)
    for r in readers:
        r.stop()
    if args.display:
        cv2.destroyAllWindows()

    for i, s in enumerate(stats):
        print(
            f"[final] cam{i}: processed={s.pose_count} avg_pose={s.avg_pose_fps:.2f} "
            f"recent_pose={s.recent_pose_fps:.2f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
