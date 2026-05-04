#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
user_site = site.getusersitepackages()
sys.path = [path for path in sys.path if path != user_site]

import cv2

from dual_rtmpose_core import (
    DEFAULT_CAMERAS,
    annotate,
    create_tracker,
    ensure_gstreamer_available,
    open_camera,
    resolve_runtime_selection,
)


@dataclass
class RecordedCamera:
    name: str
    path: str
    raw_path: Path
    overlay_path: Path
    frames: int = 0
    read_failures: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record two cameras first, then run RTMPose on the saved videos and "
            "write annotated review videos."
        )
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera path. Pass twice for two cameras. Defaults to the known by-path devices.",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--gst-decoder",
        choices=["nvjpegdec", "nvv4l2decoder", "jpegdec"],
        default="nvv4l2decoder",
    )
    parser.add_argument(
        "--mode",
        choices=["performance", "balanced", "lightweight"],
        default="balanced",
    )
    parser.add_argument("--det-frequency", type=int, default=10)
    parser.add_argument(
        "--tracking",
        dest="tracking",
        action="store_true",
        default=True,
        help="Enable rtmlib tracking between detector frames.",
    )
    parser.add_argument(
        "--no-tracking",
        dest="tracking",
        action="store_false",
        help="Disable rtmlib tracking and run detector on every processed frame.",
    )
    parser.add_argument(
        "--single-person",
        dest="single_person",
        action="store_true",
        default=True,
        help="Keep only one detected person and run pose estimation for that target only.",
    )
    parser.add_argument(
        "--multi-person",
        dest="single_person",
        action="store_false",
        help="Keep all detected persons. Useful only for debugging.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backend",
        choices=["auto", "onnxruntime", "opencv", "openvino"],
        default="auto",
    )
    parser.add_argument(
        "--trt-cache-dir",
        default="outputs/tensorrt_engines",
        help="TensorRT engine/timing cache directory used when --device tensorrt is selected.",
    )
    parser.add_argument(
        "--trt-fp16",
        dest="trt_fp16",
        action="store_true",
        default=True,
        help="Enable FP16 TensorRT engine build when --device tensorrt is selected.",
    )
    parser.add_argument(
        "--no-trt-fp16",
        dest="trt_fp16",
        action="store_false",
        help="Disable FP16 TensorRT engine build.",
    )
    parser.add_argument(
        "--trt-models",
        choices=["all", "det", "pose"],
        default="det",
        help=(
            "Apply TensorRT only to selected rtmlib models. det is the default because "
            "RTMPose on ORT TensorRT EP produced unstable keypoints in Jetson testing. "
            "all=YOLOX+RTMPose, det=YOLOX only with RTMPose on CUDA, "
            "pose=RTMPose only with YOLOX on CUDA."
        ),
    )
    parser.add_argument("--kpt-thr", type=float, default=0.4)
    parser.add_argument("--output-dir", default="outputs/recorded_rtmpose")
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC for MP4 writing. Use avc1 or H264 if your OpenCV build supports it.",
    )
    return parser.parse_args()


def make_writer(
    path: Path,
    codec: str,
    fps: int,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path} with codec={codec}")
    return writer


def build_session_dir(output_dir: str) -> Path:
    root = Path(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(100):
        suffix = "" if index == 0 else f"_{index:02d}"
        session_dir = root / f"{timestamp}{suffix}"
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return session_dir
    raise RuntimeError(f"Failed to create a unique output directory under {root}")


def record_raw_videos(
    args: argparse.Namespace,
    session_dir: Path,
) -> list[RecordedCamera]:
    camera_paths = args.cameras if args.cameras else DEFAULT_CAMERAS
    if len(camera_paths) != 2:
        raise ValueError("Pass exactly two --camera values or use the defaults.")

    ensure_gstreamer_available()
    captures = []
    writers = []
    cameras: list[RecordedCamera] = []

    try:
        for index, camera_path in enumerate(camera_paths):
            name = f"cam{index}"
            raw_path = session_dir / f"raw_{name}.mp4"
            overlay_path = session_dir / f"overlay_{name}.mp4"
            capture = open_camera(
                camera_path,
                args.width,
                args.height,
                args.fps,
                args.gst_decoder,
            )
            writer = make_writer(
                raw_path,
                args.codec,
                args.fps,
                args.width,
                args.height,
            )
            captures.append(capture)
            writers.append(writer)
            cameras.append(
                RecordedCamera(
                    name=name,
                    path=camera_path,
                    raw_path=raw_path,
                    overlay_path=overlay_path,
                )
            )

        print(f"Recording {args.seconds:.1f}s from two cameras")
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            for camera, capture, writer in zip(cameras, captures, writers):
                ok, frame = capture.read()
                if not ok or frame is None:
                    camera.read_failures += 1
                    print(f"[warn] failed to read frame from {camera.name}: {camera.path}")
                    continue
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                writer.write(frame)
                camera.frames += 1

        return cameras
    finally:
        for writer in writers:
            writer.release()
        for capture in captures:
            capture.release()


def annotate_video(
    camera: RecordedCamera,
    args: argparse.Namespace,
    *,
    backend: str,
    device: str,
    provider_options: dict[str, str] | None,
    trt_models: str,
) -> int:
    capture = cv2.VideoCapture(str(camera.raw_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open recorded video: {camera.raw_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = int(round(source_fps)) if source_fps and source_fps > 0 else args.fps
    writer = make_writer(camera.overlay_path, args.codec, output_fps, width, height)
    tracker = create_tracker(
        args.mode,
        backend,
        device,
        args.det_frequency,
        args.tracking,
        args.single_person,
        provider_options=provider_options,
        trt_models=trt_models,
    )

    processed = 0
    started_at = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            pose_started = time.perf_counter()
            keypoints, scores = tracker(frame)
            pose_ms = (time.perf_counter() - pose_started) * 1000.0
            annotated = annotate(
                frame,
                keypoints,
                scores,
                args.kpt_thr,
                camera.name,
                pose_ms,
            )
            cv2.putText(
                annotated,
                f"frame {processed + 1}",
                (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            processed += 1

            if processed % 30 == 0:
                elapsed = time.perf_counter() - started_at
                fps = processed / elapsed if elapsed > 0 else 0.0
                print(f"[pose] {camera.name}: {processed} frames, {fps:.2f} fps")
    finally:
        writer.release()
        capture.release()

    return processed


def write_side_by_side(
    cameras: list[RecordedCamera],
    session_dir: Path,
    args: argparse.Namespace,
) -> Path:
    if len(cameras) != 2:
        raise ValueError("Side-by-side preview requires exactly two cameras.")

    captures = [cv2.VideoCapture(str(camera.overlay_path)) for camera in cameras]
    try:
        for camera, capture in zip(cameras, captures):
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open overlay video: {camera.overlay_path}")

        widths = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
            for capture in captures
        ]
        heights = [
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
            for capture in captures
        ]
        fps = captures[0].get(cv2.CAP_PROP_FPS)
        output_fps = int(round(fps)) if fps and fps > 0 else args.fps
        output_height = min(heights)
        output_widths = [
            int(width * output_height / height)
            for width, height in zip(widths, heights)
        ]
        side_by_side_path = session_dir / "overlay_side_by_side.mp4"
        writer = make_writer(
            side_by_side_path,
            args.codec,
            output_fps,
            sum(output_widths),
            output_height,
        )

        try:
            while True:
                frames = []
                for capture, output_width in zip(captures, output_widths):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        return side_by_side_path
                    if frame.shape[0] != output_height or frame.shape[1] != output_width:
                        frame = cv2.resize(frame, (output_width, output_height))
                    frames.append(frame)
                writer.write(cv2.hconcat(frames))
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()


def main() -> int:
    args = parse_args()
    session_dir = build_session_dir(args.output_dir)
    runtime_selection = resolve_runtime_selection(
        args.backend,
        args.device,
        trt_cache_dir=args.trt_cache_dir,
        trt_fp16=args.trt_fp16,
        trt_models=args.trt_models,
    )

    print(f"Output directory: {session_dir}")
    print(
        f"RTMPose runtime: backend={runtime_selection.backend} "
        f"device={runtime_selection.device} provider={runtime_selection.provider}"
    )
    print(f"Runtime note: {runtime_selection.reason}")
    if runtime_selection.uses_tensorrt:
        print(
            "TensorRT: "
            f"fp16={runtime_selection.trt_fp16}, "
            f"cache={runtime_selection.trt_cache_dir}"
            f", models={runtime_selection.trt_models}"
        )

    cameras = record_raw_videos(args, session_dir)
    print("\nRecorded videos")
    for camera in cameras:
        print(
            f"  {camera.name}: {camera.frames} frames, failures={camera.read_failures}, "
            f"raw={camera.raw_path}"
        )

    print("\nRunning RTMPose on recorded videos")
    for camera in cameras:
        processed = annotate_video(
            camera,
            args,
            backend=runtime_selection.backend,
            device=runtime_selection.device,
            provider_options=runtime_selection.provider_options,
            trt_models=args.trt_models,
        )
        print(f"  {camera.name}: overlay frames={processed}, path={camera.overlay_path}")

    side_by_side_path = write_side_by_side(cameras, session_dir, args)
    print("\nReview outputs")
    for camera in cameras:
        print(f"  {camera.name} overlay: {camera.overlay_path}")
    print(f"  side-by-side: {side_by_side_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
