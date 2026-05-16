#!/usr/bin/env python3
"""Dump RTMPose TensorRT INT8 calibration blobs from recorded videos.

The output is headerless float32 NCHW data. Each sample is the exact RTMPose
input crop produced by YOLOX bbox detection plus the affine/ImageNet
normalization in pose_pipeline.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # noqa: E402
    DEFAULT_DET_MODEL,
    DEFAULT_POSE_MODEL,
    RtmposeOnnx,
    YoloxOnnx,
    select_providers,
)


DEFAULT_VIDEOS = [
    REPO_ROOT / "outputs/recorded_rtmpose/20260515_064342/raw_cam0.mp4",
    REPO_ROOT / "outputs/recorded_rtmpose/20260515_064342/raw_cam1.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump RTMPose INT8 calibration blobs from saved videos"
    )
    parser.add_argument(
        "--video",
        nargs="+",
        default=[str(p) for p in DEFAULT_VIDEOS],
        help="input MP4(s); defaults to the recorded Phase 1 dual-camera clips",
    )
    parser.add_argument(
        "--output",
        default="models/calib_rtmpose_256x192.bin",
        help="headerless float32 blob output",
    )
    parser.add_argument("--target", type=int, default=150,
                        help="target number of calibration samples")
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--det-score", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "tensorrt"],
        help="execution provider for YOLOX detection",
    )
    parser.add_argument("--trt-cache-dir", default="outputs/tensorrt_engines")
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument(
        "--trt-det",
        action="store_true",
        help="use TensorRT EP for YOLOX when --device tensorrt",
    )
    return parser.parse_args()


def sample_indices(frame_count: int, n: int, *, candidate_count: int | None = None) -> list[int]:
    if frame_count <= 0 or n <= 0:
        return []
    if candidate_count is None:
        candidate_count = n
    candidate_count = min(frame_count, max(n, candidate_count))
    if n == 1:
        return [frame_count // 2]
    raw = np.linspace(0, frame_count - 1, num=candidate_count, dtype=np.int64)
    # np.linspace can duplicate indices when candidate_count > frame_count.
    return sorted(set(int(x) for x in raw.tolist()))


def largest_bbox(dets: np.ndarray) -> np.ndarray | None:
    if dets.shape[0] == 0:
        return None
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return dets[int(np.argmax(areas))]


def dump_for_video(
    video_path: Path,
    quota: int,
    det: YoloxOnnx,
    pose: RtmposeOnnx,
    out,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate_count = int(quota * 1.5) + 16
    indices = sample_indices(total, quota, candidate_count=candidate_count)
    index_set = set(indices)
    print(
        f"[calib] {video_path}: frames={total} target={quota} sampled={len(indices)}",
        file=sys.stderr,
    )

    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx not in index_set:
            frame_idx += 1
            continue

        dets = det.infer(frame)
        bbox = largest_bbox(dets)
        if bbox is not None:
            blob, _ = pose._preprocess_one(frame, bbox)  # same code path as runtime
            out.write(np.ascontiguousarray(blob, dtype=np.float32).tobytes())
            written += 1
            if written >= quota:
                break
        else:
            print(f"[calib] no person bbox at frame {frame_idx}", file=sys.stderr)

        frame_idx += 1

    cap.release()
    print(f"[calib] {video_path}: wrote={written}", file=sys.stderr)
    return written


def main() -> int:
    args = parse_args()
    if args.target <= 0:
        print("--target must be positive", file=sys.stderr)
        return 2

    videos = [Path(p) for p in args.video]
    missing = [str(p) for p in videos if not p.exists()]
    if missing:
        print("missing video(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    det_providers = select_providers(
        args.device,
        trt_cache_dir=args.trt_cache_dir,
        trt_fp16=args.trt_fp16,
        use_trt=(args.device == "tensorrt" and args.trt_det),
    )
    print(f"[calib] detector providers={det_providers}", file=sys.stderr)
    det = YoloxOnnx(args.det_model, det_providers, score_thr=args.det_score)
    # Pose session is used only to read model input shape and reuse preprocessing.
    pose = RtmposeOnnx(args.pose_model, ["CPUExecutionProvider"])

    base = args.target // len(videos)
    rem = args.target % len(videos)
    quotas = [base + (1 if i < rem else 0) for i in range(len(videos))]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    with out_path.open("wb") as out:
        for video, quota in zip(videos, quotas):
            total_written += dump_for_video(video, quota, det, pose, out)

    sample_bytes = 3 * pose.input_h * pose.input_w * 4
    actual_bytes = out_path.stat().st_size
    print(
        f"[calib] done: samples={total_written} sample_bytes={sample_bytes} "
        f"bytes={actual_bytes} output={out_path}",
        file=sys.stderr,
    )
    if total_written == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
