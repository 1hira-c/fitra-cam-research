#!/usr/bin/env python3
"""Run the Python ONNX Runtime pipeline frame-by-frame on a recorded video
and emit per-frame YOLOX bboxes + RTMPose keypoints as JSON Lines.

Used as the numerical reference for the C++ TensorRT pipeline's correctness
check (see docs/cpp-migration-plan.md Phase 1).

Defaults:
  --device cpu              (deterministic, no CUDA / TensorRT variance)
  --det-frequency 1         (detect every frame — no caching)
  --multi-person off        (largest-bbox person only, matching default run mode)

Output schema (one JSON object per line):
  {"frame": int, "persons":
    [{"bbox": [x1, y1, x2, y2, score],
      "kpts": [[x, y, score], ...17 keypoints]}]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # noqa: E402
    DEFAULT_DET_MODEL,
    DEFAULT_POSE_MODEL,
    PoseEngine,
    RtmposeOnnx,
    YoloxOnnx,
    select_providers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True, help="input MP4")
    p.add_argument("--output", required=True, help="output JSONL path")
    p.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--det-score", type=float, default=0.5)
    p.add_argument("--det-frequency", type=int, default=1,
                   help="how often to run YOLOX (1 = every frame for correctness)")
    p.add_argument("--multi-person", action="store_true")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = whole video)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    providers = select_providers(args.device)
    det = YoloxOnnx(args.det_model, providers, score_thr=args.det_score)
    pose = RtmposeOnnx(args.pose_model, providers)
    engine = PoseEngine(
        det,
        pose,
        det_frequency=args.det_frequency,
        single_person=not args.multi_person,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"failed to open {args.video}", file=sys.stderr)
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as fout:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            result = engine.process(seq=i, frame_bgr=frame, captured_at=0.0)
            persons = []
            for (kpts, scores), bb in zip(result.persons, result.bboxes):
                persons.append({
                    "bbox": [float(v) for v in bb.tolist()],
                    "kpts": [[float(kpts[k, 0]), float(kpts[k, 1]), float(scores[k])]
                             for k in range(kpts.shape[0])],
                })
            line = {"frame": i, "persons": persons}
            fout.write(json.dumps(line, separators=(",", ":")) + "\n")
            written += 1
            i += 1
            if args.max_frames and i >= args.max_frames:
                break
            if i % 50 == 0:
                print(f"  {i}/{total or '?'} frames", file=sys.stderr, flush=True)
    cap.release()
    print(f"[done] wrote {written} frames -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
