#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
import threading
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from rtmlib import Body, PoseTracker, draw_skeleton

DEFAULT_CAMERAS = [
    "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0",
    "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0",
]

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON_EDGES = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]


@dataclass
class CameraRuntime:
    name: str
    path: str
    capture: cv2.VideoCapture
    tracker: PoseTracker
    frames: int = 0
    captured_frames: int = 0
    failures: int = 0
    published_frames: int = 0
    last_capture_ms: float = 0.0
    last_pose_ms: float = 0.0
    last_total_ms: float = 0.0
    latest_frame: Any = None
    latest_frame_index: int = 0
    latest_capture_started: float = 0.0
    latest_capture_done: float = 0.0
    inferred_frame_index: int = 0
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    latest_annotated: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class RunnerConfig:
    cameras: list[str]
    width: int = 640
    height: int = 480
    fps: int = 30
    gst_decoder: str = "nvv4l2decoder"
    mode: str = "balanced"
    det_frequency: int = 10
    tracking: bool = True
    single_person: bool = True
    device: str = "auto"
    backend: str = "auto"
    kpt_thr: float = 0.4
    max_frames: int = 0
    output_dir: str = "outputs/dual_rtmpose"
    save_every: int = 60
    display: bool = False
    publish_every: int = 1
    log_every: float = 10.0
    trt_cache_dir: str = "outputs/tensorrt_engines"
    trt_fp16: bool = True
    trt_warmup_frames: int = 0
    trt_models: str = "det"


@dataclass
class RuntimeSelection:
    backend: str
    device: str
    reason: str
    provider: str
    trt_cache_dir: str | None = None
    trt_fp16: bool | None = None
    trt_models: str | None = None
    provider_options: dict[str, str] | None = None

    @property
    def uses_tensorrt(self) -> bool:
        return self.provider == "tensorrt"


def add_common_camera_args(parser, *, include_display: bool) -> None:
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera path. Pass twice for two cameras. Defaults to the two known by-path devices.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--gst-decoder",
        choices=["nvjpegdec", "nvv4l2decoder", "jpegdec"],
        default="nvv4l2decoder",
        help="GStreamer JPEG decoder. nvv4l2decoder is the default in this Jetson setup.",
    )
    parser.add_argument(
        "--mode",
        choices=["performance", "balanced", "lightweight"],
        default="balanced",
        help="rtmlib model preset. balanced is the default Jetson 30fps target.",
    )
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=10,
        help="Run person detection every N frames and track in-between. Higher values reduce detector load.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "rtmlib device, e.g. auto/cpu/cuda/cuda:0/tensorrt. "
            "auto prefers CUDA and does not auto-select TensorRT."
        ),
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "onnxruntime", "opencv", "openvino"],
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.4,
        help="Keypoint score threshold for drawing and Web rendering.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames per camera. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dual_rtmpose",
        help="Directory to save snapshots.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=60,
        help="Save one annotated frame every N processed frames per camera. 0 disables saving.",
    )
    parser.add_argument(
        "--publish-every",
        type=int,
        default=1,
        help="Publish one pose payload every N processed frames per camera.",
    )
    parser.add_argument(
        "--log-every",
        type=float,
        default=10.0,
        help="Print per-camera progress every N seconds. 0 disables periodic logs.",
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
        "--trt-warmup-frames",
        type=int,
        default=0,
        help=(
            "Keep runner status in TensorRT build/warmup mode until each camera has "
            "processed N frames. 0 disables explicit warmup status."
        ),
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
    if include_display:
        parser.add_argument(
            "--display",
            action="store_true",
            help="Show live windows. Leave off on headless setups.",
        )


def build_runner_config(args, *, allow_display: bool) -> RunnerConfig:
    cameras = args.cameras if args.cameras else DEFAULT_CAMERAS
    if len(cameras) != 2:
        raise ValueError("Pass exactly two --camera values or use the defaults.")
    if args.publish_every < 1:
        raise ValueError("--publish-every must be >= 1.")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0.")
    if args.trt_warmup_frames < 0:
        raise ValueError("--trt-warmup-frames must be >= 0.")

    return RunnerConfig(
        cameras=list(cameras),
        width=args.width,
        height=args.height,
        fps=args.fps,
        gst_decoder=args.gst_decoder,
        mode=args.mode,
        det_frequency=args.det_frequency,
        tracking=args.tracking,
        single_person=args.single_person,
        device=args.device,
        backend=args.backend,
        kpt_thr=args.kpt_thr,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        save_every=args.save_every,
        display=getattr(args, "display", False) if allow_display else False,
        publish_every=args.publish_every,
        log_every=args.log_every,
        trt_cache_dir=args.trt_cache_dir,
        trt_fp16=args.trt_fp16,
        trt_warmup_frames=args.trt_warmup_frames,
        trt_models=args.trt_models,
    )


def ensure_gstreamer_available() -> None:
    build_info = cv2.getBuildInformation()
    for line in build_info.splitlines():
        if "GStreamer:" not in line:
            continue
        if "YES" in line:
            return
        break

    raise RuntimeError(
        "This OpenCV build does not have GStreamer enabled. "
        f"Current cv2 module: {cv2.__file__}. "
        "Use the apt-provided python3-opencv build and remove pip-installed OpenCV wheels "
        "from the active virtual environment."
    )


def get_onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def get_opencv_cuda_device_count() -> int:
    if not hasattr(cv2, "cuda"):
        return 0
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except cv2.error:
        return 0


def build_tensorrt_provider_options(
    cache_dir: str,
    *,
    fp16: bool,
    device: str = "tensorrt",
) -> dict[str, str]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    device_id = "0"
    if ":" in device:
        device_id = device.rsplit(":", 1)[-1]

    return {
        "device_id": device_id,
        "trt_fp16_enable": "True" if fp16 else "False",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(cache_path),
        "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": str(cache_path),
    }


def resolve_runtime_selection(
    backend: str,
    device: str,
    *,
    trt_cache_dir: str = "outputs/tensorrt_engines",
    trt_fp16: bool = True,
    trt_models: str = "det",
) -> RuntimeSelection:
    ort_providers = get_onnxruntime_providers()
    has_ort_cuda = "CUDAExecutionProvider" in ort_providers
    has_ort_trt = "TensorrtExecutionProvider" in ort_providers
    opencv_cuda_count = get_opencv_cuda_device_count()

    def cpu_selection(selected_backend: str, reason: str) -> RuntimeSelection:
        return RuntimeSelection(
            backend=selected_backend,
            device="cpu",
            provider="cpu",
            reason=reason,
        )

    if device.startswith("tensorrt"):
        if backend not in ("auto", "onnxruntime"):
            raise RuntimeError(
                "TensorRT execution requires --backend auto or --backend onnxruntime."
            )
        if not has_ort_trt:
            raise RuntimeError(
                "Requested TensorRT execution, but TensorrtExecutionProvider is not installed. "
                f"Available onnxruntime providers: {ort_providers or 'none'}. "
                "Install the JetPack-compatible onnxruntime-gpu wheel or pass --device cuda/cpu."
            )
        if not has_ort_cuda:
            raise RuntimeError(
                "Requested TensorRT execution, but CUDAExecutionProvider is not installed. "
                "TensorRT provider fallback requires CUDAExecutionProvider in this prototype."
            )
        provider_options = build_tensorrt_provider_options(
            trt_cache_dir,
            fp16=trt_fp16,
            device=device,
        )
        return RuntimeSelection(
            backend="onnxruntime",
            device="tensorrt",
            provider="tensorrt",
            trt_cache_dir=trt_cache_dir,
            trt_fp16=trt_fp16,
            trt_models=trt_models,
            provider_options=provider_options,
            reason=(
                "requested TensorRT Execution Provider with CUDA/CPU fallback; "
                f"first launch may build TensorRT engines; trt_models={trt_models}"
            ),
        )

    if backend == "auto" and device == "auto":
        if has_ort_cuda:
            return RuntimeSelection(
                backend="onnxruntime",
                device="cuda",
                provider="cuda",
                reason="auto-selected onnxruntime CUDAExecutionProvider",
            )
        if opencv_cuda_count > 0:
            return RuntimeSelection(
                backend="opencv",
                device="cuda",
                provider="opencv-cuda",
                reason="auto-selected OpenCV CUDA backend",
            )
        return cpu_selection(
            "onnxruntime",
            "auto-selected CPU fallback because no CUDAExecutionProvider or OpenCV CUDA device was available",
        )

    if backend == "auto":
        if device.startswith("cuda"):
            if has_ort_cuda:
                return RuntimeSelection(
                    backend="onnxruntime",
                    device=device,
                    provider="cuda",
                    reason="requested CUDA device, using onnxruntime CUDAExecutionProvider",
                )
            if opencv_cuda_count > 0 and device == "cuda":
                return RuntimeSelection(
                    backend="opencv",
                    device="cuda",
                    provider="opencv-cuda",
                    reason="requested CUDA device, using OpenCV CUDA backend",
                )
            raise RuntimeError(
                "CUDA was requested, but this environment does not expose onnxruntime CUDAExecutionProvider "
                "or an OpenCV CUDA device. Install a CUDA-capable runtime on Jetson or pass --device cpu."
            )
        if device == "cpu":
            return cpu_selection("onnxruntime", "requested CPU device")
        return RuntimeSelection(
            backend="openvino",
            device=device,
            provider="openvino",
            reason="requested non-default device with automatic backend selection",
        )

    if device == "auto":
        if backend == "onnxruntime":
            if has_ort_cuda:
                return RuntimeSelection(
                    backend="onnxruntime",
                    device="cuda",
                    provider="cuda",
                    reason="auto-selected CUDAExecutionProvider for onnxruntime",
                )
            return cpu_selection(
                "onnxruntime",
                "falling back to CPUExecutionProvider because CUDAExecutionProvider was unavailable",
            )
        if backend == "opencv":
            if opencv_cuda_count > 0:
                return RuntimeSelection(
                    backend="opencv",
                    device="cuda",
                    provider="opencv-cuda",
                    reason="auto-selected OpenCV CUDA backend",
                )
            return cpu_selection(
                "opencv",
                "falling back to OpenCV CPU target because no CUDA device was available",
            )
        return cpu_selection("openvino", "OpenVINO runtime uses CPU mode in this prototype")

    if backend == "onnxruntime" and device.startswith("cuda") and not has_ort_cuda:
        raise RuntimeError(
            "Requested onnxruntime CUDA execution, but CUDAExecutionProvider is not installed. "
            "Install a CUDA-capable onnxruntime build on Jetson or pass --device cpu."
        )

    if backend == "opencv" and device == "cuda" and opencv_cuda_count < 1:
        raise RuntimeError(
            "Requested OpenCV CUDA execution, but no CUDA-enabled OpenCV device was available. "
            "Use a CUDA-enabled OpenCV build on Jetson or pass --device cpu."
        )

    if backend == "openvino" and device != "cpu":
        return cpu_selection("openvino", "OpenVINO backend only supports CPU in rtmlib")

    return RuntimeSelection(
        backend=backend,
        device=device,
        provider=device if backend == "onnxruntime" else backend,
        reason="using explicitly requested runtime",
    )


def quote_gst_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_camera_pipeline(
    path: str, width: int, height: int, fps: int, gst_decoder: str
) -> str:
    resolved_path = str(Path(path).resolve())
    source = (
        f'v4l2src device="{quote_gst_string(resolved_path)}" do-timestamp=true ! '
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
    )
    if gst_decoder == "nvjpegdec":
        decode = "jpegparse ! nvjpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    elif gst_decoder == "nvv4l2decoder":
        decode = (
            "jpegparse ! nvv4l2decoder mjpeg=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
        )
    else:
        decode = "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "

    sink = "queue leaky=downstream max-size-buffers=1 ! appsink drop=true max-buffers=1 sync=false"
    return source + decode + sink


def open_camera(
    path: str, width: int, height: int, fps: int, gst_decoder: str
) -> cv2.VideoCapture:
    pipeline = build_camera_pipeline(path, width, height, fps, gst_decoder)
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise RuntimeError(
            f"Failed to open camera via GStreamer: {path}\n"
            f"Pipeline: {pipeline}\n"
            "Try --gst-decoder nvv4l2decoder if nvjpegdec is not accepted by this camera."
        )
    return capture


@contextmanager
def patch_rtmlib_tensorrt(provider_options: dict[str, str], *, trt_models: str):
    from rtmlib.tools import base as base_module

    original_init = base_module.BaseTool.__init__

    trt_class_names = {
        "all": {"YOLOX", "RTMPose"},
        "det": {"YOLOX"},
        "pose": {"RTMPose"},
    }[trt_models]

    def patched_init(
        self,
        onnx_model: str | None = None,
        model_input_size: tuple | None = None,
        mean: tuple | None = None,
        std: tuple | None = None,
        backend: str = "opencv",
        device: str = "cpu",
    ):
        if backend != "onnxruntime" or device != "tensorrt":
            return original_init(
                self,
                onnx_model,
                model_input_size,
                mean,
                std,
                backend,
                device,
            )

        class_name = type(self).__name__
        if class_name not in trt_class_names:
            return original_init(
                self,
                onnx_model,
                model_input_size,
                mean,
                std,
                backend,
                "cuda",
            )

        if not base_module.os.path.exists(onnx_model):
            onnx_model = base_module.download_checkpoint(onnx_model)

        import onnxruntime as ort

        providers = [
            ("TensorrtExecutionProvider", dict(provider_options)),
            ("CUDAExecutionProvider", {"device_id": provider_options["device_id"]}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(
            path_or_bytes=onnx_model,
            providers=providers,
        )
        active_providers = self.session.get_providers()
        if "TensorrtExecutionProvider" not in active_providers:
            raise RuntimeError(
                "TensorRT Execution Provider was requested but the ONNX Runtime "
                f"session did not enable it for {onnx_model}. "
                f"Active providers: {active_providers}"
            )
        print(
            f"load {onnx_model} with onnxruntime TensorRT backend "
            f"(providers={active_providers})"
        )

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    base_module.BaseTool.__init__ = patched_init
    try:
        yield
    finally:
        base_module.BaseTool.__init__ = original_init


def _bbox_area(bbox) -> float:
    try:
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        return width * height
    except (IndexError, TypeError, ValueError):
        return 0.0


def _select_largest_bbox_index(bboxes) -> int | None:
    if bboxes is None or len(bboxes) == 0:
        return None
    return max(range(len(bboxes)), key=lambda index: _bbox_area(bboxes[index]))


def _take_single_item(values, index: int):
    if isinstance(values, np.ndarray):
        return values[index : index + 1]
    return [values[index]]


class SinglePersonDetector:
    def __init__(self, detector) -> None:
        self.detector = detector
        self.mode = getattr(detector, "mode", None)

    def __call__(self, image):
        result = self.detector(image)
        if isinstance(result, tuple):
            bboxes, extra = result
            index = _select_largest_bbox_index(bboxes)
            if index is None:
                return result
            return _take_single_item(bboxes, index), _take_single_item(extra, index)

        index = _select_largest_bbox_index(result)
        if index is None:
            return result
        return _take_single_item(result, index)


def _person_score(score_set) -> float:
    finite_scores = [float(score) for score in score_set if math.isfinite(float(score))]
    if not finite_scores:
        return 0.0
    return sum(finite_scores) / len(finite_scores)


def keep_best_person(keypoints, scores):
    if len(keypoints) <= 1:
        return keypoints, scores

    best_index = max(range(len(scores)), key=lambda index: _person_score(scores[index]))
    return _take_single_item(keypoints, best_index), _take_single_item(scores, best_index)


def create_tracker(
    mode: str,
    backend: str,
    device: str,
    det_frequency: int,
    tracking: bool,
    single_person: bool,
    *,
    provider_options: dict[str, str] | None = None,
    trt_models: str = "det",
) -> PoseTracker:
    if device == "tensorrt":
        if provider_options is None:
            raise RuntimeError("TensorRT provider options were not configured.")
        with patch_rtmlib_tensorrt(provider_options, trt_models=trt_models):
            tracker = PoseTracker(
                Body,
                mode=mode,
                det_frequency=det_frequency,
                tracking=tracking,
                backend=backend,
                device=device,
                to_openpose=False,
            )
        if single_person and tracker.det_model is not None:
            tracker.det_model = SinglePersonDetector(tracker.det_model)
        return tracker

    tracker = PoseTracker(
        Body,
        mode=mode,
        det_frequency=det_frequency,
        tracking=tracking,
        backend=backend,
        device=device,
        to_openpose=False,
    )
    if single_person and tracker.det_model is not None:
        tracker.det_model = SinglePersonDetector(tracker.det_model)
    return tracker


def runtime_metadata(runtime_selection: RuntimeSelection) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "backend": runtime_selection.backend,
        "device": runtime_selection.device,
        "provider": runtime_selection.provider,
    }
    if runtime_selection.uses_tensorrt:
        metadata["trt_fp16"] = runtime_selection.trt_fp16
        metadata["trt_cache_dir"] = runtime_selection.trt_cache_dir
        metadata["trt_models"] = runtime_selection.trt_models
    return metadata


def annotate(frame, keypoints, scores, kpt_thr: float, name: str, elapsed_ms: float):
    annotated = frame.copy()
    annotated = draw_skeleton(
        annotated,
        keypoints,
        scores,
        kpt_thr=kpt_thr,
        openpose_skeleton=False,
    )
    cv2.putText(
        annotated,
        f"{name}  {elapsed_ms:.1f} ms",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def calculate_recent_fps(frame_times: deque[float]) -> float:
    if len(frame_times) < 2:
        return 0.0
    elapsed_sec = frame_times[-1] - frame_times[0]
    if elapsed_sec <= 0:
        return 0.0
    return (len(frame_times) - 1) / elapsed_sec


def _serialize_person(keypoint_set, score_set, *, score_threshold: float) -> dict[str, Any]:
    keypoints: list[list[float]] = []
    valid_points: list[tuple[float, float]] = []
    visible_points: list[tuple[float, float]] = []
    finite_scores: list[float] = []

    for point, score in zip(keypoint_set, score_set):
        x = float(point[0])
        y = float(point[1])
        confidence = float(score)
        keypoints.append([x, y, confidence])
        if math.isfinite(confidence):
            finite_scores.append(confidence)
        if math.isfinite(x) and math.isfinite(y) and confidence > 0.0:
            valid_points.append((x, y))
        if math.isfinite(x) and math.isfinite(y) and confidence >= score_threshold:
            visible_points.append((x, y))

    bbox = None
    if valid_points:
        xs = [point[0] for point in valid_points]
        ys = [point[1] for point in valid_points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

    return {
        "keypoints2d": keypoints,
        "bbox": bbox,
        "keypoint_stats": {
            "count": len(keypoints),
            "positive_count": len(valid_points),
            "visible_count": len(visible_points),
            "max_score": max(finite_scores) if finite_scores else 0.0,
            "min_score": min(finite_scores) if finite_scores else 0.0,
        },
    }


def build_pose_payload(
    runtime: CameraRuntime,
    *,
    runtime_selection: RuntimeSelection,
    kpt_thr: float,
    frame,
    keypoints,
    scores,
    capture_ms: float,
    pose_ms: float,
    total_ms: float,
    started_at: float,
) -> dict[str, Any]:
    persons = []
    for index, (keypoint_set, score_set) in enumerate(zip(keypoints, scores)):
        person = _serialize_person(keypoint_set, score_set, score_threshold=kpt_thr)
        person["id"] = index
        persons.append(person)

    elapsed_sec = time.perf_counter() - started_at
    with runtime.lock:
        frames = runtime.frames
        captured_frames = runtime.captured_frames
        failures = runtime.failures
        published_frames = runtime.published_frames
        recent_fps = calculate_recent_fps(runtime.frame_times)
    pending_frames = max(0, captured_frames - frames)

    avg_fps = frames / elapsed_sec if elapsed_sec > 0 else 0.0
    avg_publish_fps = published_frames / elapsed_sec if elapsed_sec > 0 else 0.0

    return {
        "camera_id": runtime.name,
        "camera_path": runtime.path,
        "sequence": frames,
        "server_publish_ts": time.time(),
        "runtime": runtime_metadata(runtime_selection),
        "frame_size": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        },
        "stage_ms": {
            "capture": capture_ms,
            "pose": pose_ms,
            "total": total_ms,
        },
        "metrics": {
            "frames": frames,
            "captured_frames": captured_frames,
            "pending_frames": pending_frames,
            "dropped_frames": pending_frames,
            "failures": failures,
            "published_frames": published_frames,
            "avg_fps": avg_fps,
            "avg_publish_fps": avg_publish_fps,
            "recent_fps": recent_fps,
        },
        "persons": persons,
        "skeleton": {
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "edges": COCO_SKELETON_EDGES,
            "score_threshold": kpt_thr,
        },
    }


def build_summary(
    runtimes: list[CameraRuntime],
    total_sec: float,
    runtime_selection: RuntimeSelection,
) -> dict[str, Any]:
    cameras = []
    for runtime in runtimes:
        with runtime.lock:
            frames = runtime.frames
            captured_frames = runtime.captured_frames
            failures = runtime.failures
            published_frames = runtime.published_frames
            recent_fps = calculate_recent_fps(runtime.frame_times)
            last_capture_ms = runtime.last_capture_ms
            last_pose_ms = runtime.last_pose_ms
            last_total_ms = runtime.last_total_ms

        pending_frames = max(0, captured_frames - frames)
        avg_fps = frames / total_sec if total_sec > 0 else 0.0
        avg_publish_fps = published_frames / total_sec if total_sec > 0 else 0.0
        cameras.append(
            {
                "camera_id": runtime.name,
                "camera_path": runtime.path,
                "frames": frames,
                "captured_frames": captured_frames,
                "pending_frames": pending_frames,
                "dropped_frames": pending_frames,
                "failures": failures,
                "published_frames": published_frames,
                "avg_fps": avg_fps,
                "avg_publish_fps": avg_publish_fps,
                "recent_fps": recent_fps,
                "last_stage_ms": {
                    "capture": last_capture_ms,
                    "pose": last_pose_ms,
                    "total": last_total_ms,
                },
            }
        )

    return {
        "total_runtime_sec": total_sec,
        "runtime": {
            **runtime_metadata(runtime_selection),
            "reason": runtime_selection.reason,
        },
        "cameras": cameras,
    }


class PoseSnapshotStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle_sequence = 0
        self._camera_payloads: dict[str, dict[str, Any]] = {}
        self._bundle: dict[str, Any] = {
            "type": "pose_bundle",
            "bundle_sequence": self._bundle_sequence,
            "server_publish_ts": time.time(),
            "runner_status": "idle",
            "runner_message": "waiting to start",
            "cameras": [],
            "summary": None,
        }

    def _refresh_bundle_locked(self) -> None:
        self._bundle_sequence += 1
        self._bundle = {
            "type": "pose_bundle",
            "bundle_sequence": self._bundle_sequence,
            "server_publish_ts": time.time(),
            "runner_status": self._bundle["runner_status"],
            "runner_message": self._bundle["runner_message"],
            "cameras": [
                self._camera_payloads[key] for key in sorted(self._camera_payloads)
            ],
            "summary": self._bundle["summary"],
        }

    def set_runner_state(
        self,
        status: str,
        message: str,
        *,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._bundle["runner_status"] = status
            self._bundle["runner_message"] = message
            self._bundle["summary"] = summary
            self._refresh_bundle_locked()

    def publish_camera_payload(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._camera_payloads[payload["camera_id"]] = copy.deepcopy(payload)
            self._refresh_bundle_locked()

    def get_bundle(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._bundle)


class DualCameraPoseRunner:
    def __init__(
        self,
        config: RunnerConfig,
        *,
        publisher: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.publisher = publisher
        self.runtimes: list[CameraRuntime] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera_threads: list[threading.Thread] = []
        self.worker_errors: list[Exception] = []
        self.worker_error_lock = threading.Lock()
        self.runtime_selection = resolve_runtime_selection(
            config.backend,
            config.device,
            trt_cache_dir=config.trt_cache_dir,
            trt_fp16=config.trt_fp16,
            trt_models=config.trt_models,
        )

    def _open_runtimes(self) -> None:
        ensure_gstreamer_available()
        self.runtimes = []
        for index, path in enumerate(self.config.cameras):
            capture = open_camera(
                path,
                self.config.width,
                self.config.height,
                self.config.fps,
                self.config.gst_decoder,
            )
            tracker = create_tracker(
                self.config.mode,
                self.runtime_selection.backend,
                self.runtime_selection.device,
                self.config.det_frequency,
                self.config.tracking,
                self.config.single_person,
                provider_options=self.runtime_selection.provider_options,
                trt_models=self.config.trt_models,
            )
            self.runtimes.append(
                CameraRuntime(
                    name=f"cam{index}",
                    path=path,
                    capture=capture,
                    tracker=tracker,
                )
            )

    def _cleanup(self) -> None:
        for runtime in self.runtimes:
            runtime.capture.release()
        if self.config.display:
            cv2.destroyAllWindows()

    def _record_worker_error(self, exc: Exception) -> None:
        with self.worker_error_lock:
            self.worker_errors.append(exc)

    def _capture_camera(self, runtime: CameraRuntime) -> None:
        try:
            while not self.stop_event.is_set():
                with runtime.lock:
                    if (
                        self.config.max_frames
                        and runtime.frames >= self.config.max_frames
                    ):
                        break
                    has_pending_frame = (
                        runtime.latest_frame is not None
                        and runtime.latest_frame_index != runtime.inferred_frame_index
                    )
                if has_pending_frame:
                    time.sleep(0.002)
                    continue

                capture_started = time.perf_counter()
                ok, frame = runtime.capture.read()
                capture_done = time.perf_counter()
                if not ok or frame is None:
                    with runtime.lock:
                        runtime.failures += 1
                    print(
                        f"[warn] failed to read frame from {runtime.name} ({runtime.path})"
                    )
                    continue

                with runtime.lock:
                    runtime.captured_frames += 1
                    runtime.latest_frame_index = runtime.captured_frames
                    runtime.latest_frame = frame
                    runtime.latest_capture_started = capture_started
                    runtime.latest_capture_done = capture_done
                    runtime.last_capture_ms = (
                        capture_done - capture_started
                    ) * 1000.0
        except Exception as exc:
            self._record_worker_error(exc)
            self.stop_event.set()

    def _start_camera_threads(self) -> None:
        self.camera_threads = []
        self.worker_errors = []
        for runtime in self.runtimes:
            thread = threading.Thread(
                target=self._capture_camera,
                args=(runtime,),
                name=f"dual-rtmpose-capture-{runtime.name}",
                daemon=True,
            )
            self.camera_threads.append(thread)
            thread.start()

    def _join_camera_threads(self, timeout: float | None = None) -> None:
        for thread in self.camera_threads:
            thread.join(timeout=timeout)

    def _show_display_frames(self) -> bool:
        for runtime in self.runtimes:
            with runtime.lock:
                annotated = runtime.latest_annotated
            if annotated is not None:
                cv2.imshow(runtime.name, annotated)

        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def _get_latest_frame(self, runtime: CameraRuntime):
        with runtime.lock:
            if self.config.max_frames and runtime.frames >= self.config.max_frames:
                return None
            if runtime.latest_frame is None:
                return None
            if runtime.latest_frame_index == runtime.inferred_frame_index:
                return None
            return (
                runtime.latest_frame,
                runtime.latest_frame_index,
                runtime.latest_capture_started,
                runtime.latest_capture_done,
            )

    def _all_cameras_done(self) -> bool:
        if not self.config.max_frames:
            return False
        for runtime in self.runtimes:
            with runtime.lock:
                if runtime.frames < self.config.max_frames:
                    return False
        return True

    def _process_latest_frame(
        self,
        runtime: CameraRuntime,
        *,
        output_dir: Path,
        started_at: float,
        status_store: PoseSnapshotStore | None = None,
    ) -> bool:
        frame_bundle = self._get_latest_frame(runtime)
        if frame_bundle is None:
            return False

        frame, frame_index, capture_started, capture_done = frame_bundle
        is_first_tensorrt_frame = (
            self.runtime_selection.uses_tensorrt and runtime.frames == 0
        )
        if is_first_tensorrt_frame:
            message = (
                f"building TensorRT engines during first inference for {runtime.name}; "
                "this can take several minutes on first cache creation"
            )
            print(f"[trt] {message}")
            if status_store is not None:
                status_store.set_runner_state("starting", message)

        pose_started = time.perf_counter()
        keypoints, scores = runtime.tracker(frame)
        if self.config.single_person:
            keypoints, scores = keep_best_person(keypoints, scores)
        pose_done = time.perf_counter()
        if is_first_tensorrt_frame:
            print(
                f"[trt] first inference for {runtime.name} completed in "
                f"{(pose_done - pose_started):.1f}s"
            )

        capture_ms = (capture_done - capture_started) * 1000.0
        pose_ms = (pose_done - capture_done) * 1000.0
        total_ms = (pose_done - capture_started) * 1000.0

        with runtime.lock:
            if frame_index <= runtime.inferred_frame_index:
                return False
            runtime.inferred_frame_index = frame_index
            runtime.frames += 1
            frame_number = runtime.frames
            runtime.last_capture_ms = capture_ms
            runtime.last_pose_ms = pose_ms
            runtime.last_total_ms = total_ms
            runtime.frame_times.append(pose_done)

        should_annotate = self.config.display or (
            self.config.save_every and frame_number % self.config.save_every == 0
        )
        annotated = None
        if should_annotate:
            annotated = annotate(
                frame,
                keypoints,
                scores,
                self.config.kpt_thr,
                runtime.name,
                pose_ms,
            )

        if self.config.display and annotated is not None:
            with runtime.lock:
                runtime.latest_annotated = annotated

        if (
            self.config.save_every
            and frame_number % self.config.save_every == 0
            and annotated is not None
        ):
            out_path = output_dir / f"{runtime.name}_{frame_number:06d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"[saved] {out_path}")

        if self.publisher is None or frame_number % self.config.publish_every != 0:
            return True

        with runtime.lock:
            runtime.published_frames += 1
        payload = build_pose_payload(
            runtime,
            runtime_selection=self.runtime_selection,
            kpt_thr=self.config.kpt_thr,
            frame=frame,
            keypoints=keypoints,
            scores=scores,
            capture_ms=capture_ms,
            pose_ms=pose_ms,
            total_ms=total_ms,
            started_at=started_at,
        )
        self.publisher(payload)
        return True

    def _print_progress(self, started_at: float) -> None:
        elapsed_sec = time.perf_counter() - started_at
        if elapsed_sec <= 0:
            return

        parts = []
        for runtime in self.runtimes:
            with runtime.lock:
                frames = runtime.frames
                captured_frames = runtime.captured_frames
                failures = runtime.failures
                recent_fps = calculate_recent_fps(runtime.frame_times)
                last_total_ms = runtime.last_total_ms
            avg_fps = frames / elapsed_sec
            pending_frames = max(0, captured_frames - frames)
            parts.append(
                f"{runtime.name}: avg={avg_fps:.1f} recent={recent_fps:.1f} "
                f"stage={last_total_ms:.1f}ms pending={pending_frames} failures={failures}"
            )
        print(f"[progress] {elapsed_sec:.0f}s " + " | ".join(parts))

    def run(self, *, status_store: PoseSnapshotStore | None = None) -> dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        if self.config.save_every:
            output_dir.mkdir(parents=True, exist_ok=True)

        started_at = time.perf_counter()
        try:
            self.stop_event.clear()
            if status_store is not None:
                start_message = "opening cameras"
                if self.runtime_selection.uses_tensorrt:
                    start_message = (
                        "building TensorRT engines; first launch may take several minutes"
                    )
                status_store.set_runner_state("starting", start_message)

            self._open_runtimes()

            print("Started dual-camera RTMPose")
            print(f"  OpenCV: {cv2.__version__} ({cv2.__file__})")
            print(f"  decoder: {self.config.gst_decoder}")
            print(
                f"  runtime: backend={self.runtime_selection.backend} "
                f"device={self.runtime_selection.device} "
                f"provider={self.runtime_selection.provider}"
            )
            print(f"  runtime note: {self.runtime_selection.reason}")
            if self.runtime_selection.uses_tensorrt:
                print(f"  TensorRT FP16: {self.runtime_selection.trt_fp16}")
                print(f"  TensorRT cache: {self.runtime_selection.trt_cache_dir}")
                print(f"  TensorRT models: {self.runtime_selection.trt_models}")
            print(f"  tracking: {self.config.tracking}")
            print(f"  single person: {self.config.single_person}")
            for runtime in self.runtimes:
                print(f"  {runtime.name}: {runtime.path}")

            started_at = time.perf_counter()
            next_log_at = started_at + self.config.log_every
            warmup_active = (
                self.runtime_selection.uses_tensorrt
                and max(1, self.config.trt_warmup_frames) > 0
            )
            warmup_frames = max(1, self.config.trt_warmup_frames)
            if status_store is not None and warmup_active:
                status_store.set_runner_state(
                    "starting",
                    "building TensorRT engines during first inference",
                )
            if status_store is not None and not warmup_active:
                status_store.set_runner_state(
                    "running", "capturing and publishing 2D skeletons"
                )

            self._start_camera_threads()
            while any(thread.is_alive() for thread in self.camera_threads):
                if self.config.display and self._show_display_frames():
                    self.stop_event.set()
                    break

                with self.worker_error_lock:
                    has_worker_error = bool(self.worker_errors)
                if has_worker_error:
                    self.stop_event.set()
                    break

                did_process = False
                for runtime in self.runtimes:
                    did_process = (
                        self._process_latest_frame(
                            runtime,
                            output_dir=output_dir,
                            started_at=started_at,
                            status_store=status_store,
                        )
                        or did_process
                    )

                if warmup_active and all(
                    runtime.frames >= warmup_frames
                    for runtime in self.runtimes
                ):
                    warmup_active = False
                    if status_store is not None:
                        status_store.set_runner_state(
                            "running",
                            "capturing and publishing 2D skeletons",
                        )

                if self._all_cameras_done():
                    self.stop_event.set()
                    break

                if (
                    self.config.log_every > 0
                    and time.perf_counter() >= next_log_at
                ):
                    self._print_progress(started_at)
                    next_log_at += self.config.log_every

                if not did_process:
                    time.sleep(0.002)

            self._join_camera_threads()
            with self.worker_error_lock:
                if self.worker_errors:
                    raise RuntimeError("Camera worker failed") from self.worker_errors[0]

            total_sec = time.perf_counter() - started_at
            summary = build_summary(
                self.runtimes,
                total_sec,
                self.runtime_selection,
            )
            print("\nSummary")
            for camera in summary["cameras"]:
                print(
                    f"  {camera['camera_id']}: frames={camera['frames']}, "
                    f"failures={camera['failures']}, avg_fps={camera['avg_fps']:.2f}, "
                    f"recent_fps={camera['recent_fps']:.2f}, "
                    f"publish_fps={camera['avg_publish_fps']:.2f}, "
                    f"last_total_ms={camera['last_stage_ms']['total']:.1f}"
                )
            if status_store is not None:
                status_store.set_runner_state(
                    "stopped", "runner stopped", summary=summary
                )
            return summary
        finally:
            self.stop_event.set()
            self._join_camera_threads(timeout=1.0)
            self._cleanup()

    def _thread_target(self, status_store: PoseSnapshotStore) -> None:
        try:
            self.run(status_store=status_store)
        except Exception as exc:
            status_store.set_runner_state("error", str(exc))
            raise

    def start_background(self, status_store: PoseSnapshotStore) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Runner is already active.")

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._thread_target,
            args=(status_store,),
            name="dual-rtmpose-runner",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
