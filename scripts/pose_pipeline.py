"""Shared runtime for fitra-cam.

V4L2 capture + ONNX YOLOX person detector + ONNX RTMPose SimCC keypoint
regressor. No rtmlib dependency. The default ONNX paths point at the
weights distributed by rtmlib's release archive, which we treat as
already-downloaded artifacts.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


COCO_KP_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

COCO_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

CAM_COLORS = (
    (0, 220, 0),
    (0, 180, 255),
)

DEFAULT_DET_MODEL = str(
    Path.home() / ".cache/rtmlib/hub/checkpoints/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
)
DEFAULT_POSE_MODEL = str(
    Path.home() / ".cache/rtmlib/hub/checkpoints/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"
)

DEFAULT_CAM0 = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
DEFAULT_CAM1 = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"

IMAGENET_MEAN_BGR = np.array([103.53, 116.28, 123.675], dtype=np.float32)
IMAGENET_STD_BGR = np.array([57.375, 57.12, 58.395], dtype=np.float32)


# ---------- provider selection ----------

def select_providers(
    device: str,
    *,
    trt_cache_dir: Optional[str] = None,
    trt_fp16: bool = False,
    use_trt: bool = False,
):
    """Return a providers list ready to pass into InferenceSession.

    `device` chooses the base accelerator; `use_trt` (combined with
    `device == "tensorrt"`) toggles whether THIS specific model should
    actually run on TensorRT. When `use_trt` is False under tensorrt
    mode we fall back to CUDA so the caller can mix providers per model.
    """
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider not available; install Jetson AI Lab onnxruntime-gpu wheel"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "tensorrt":
        if use_trt:
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError("TensorrtExecutionProvider not available")
            opts: dict = {}
            if trt_cache_dir:
                Path(trt_cache_dir).mkdir(parents=True, exist_ok=True)
                opts["trt_engine_cache_enable"] = True
                opts["trt_engine_cache_path"] = trt_cache_dir
            if trt_fp16:
                opts["trt_fp16_enable"] = True
            return [
                ("TensorrtExecutionProvider", opts),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    # auto
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------- V4L2 capture ----------

@dataclasses.dataclass
class CameraConfig:
    path: str
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "MJPG"


def open_v4l2(cfg: CameraConfig) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cfg.path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open {cfg.path} with CAP_V4L2")
    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    # OpenCV V4L2 needs >=2 buffers to pipeline queue/dequeue; with BUFFERSIZE=1
    # each grab waits a full frame period (≈60ms at 640x480 MJPG on this UVC
    # camera, capping us at ~15 fps regardless of the camera's true rate).
    # CameraReader already keeps only the latest frame at the Python layer,
    # so the small V4L2-side queue does not translate into perceived latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


class CameraReader(threading.Thread):
    """Threaded reader that keeps only the most recent frame."""

    def __init__(self, cam_id: int, cfg: CameraConfig):
        super().__init__(daemon=True, name=f"CameraReader-{cam_id}")
        self.cam_id = cam_id
        self.cfg = cfg
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[tuple[int, np.ndarray, float]] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self.seq = 0
        self._recv_times: deque[float] = deque(maxlen=60)
        self.recv_fps = 0.0

    def start(self) -> "CameraReader":
        self.cap = open_v4l2(self.cfg)
        super().start()
        return self

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self) -> None:
        assert self.cap is not None
        while not self._stop_evt.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            now = time.monotonic()
            self.seq += 1
            with self._lock:
                self._latest = (self.seq, frame, now)
                self._recv_times.append(now)
                if len(self._recv_times) >= 2:
                    span = self._recv_times[-1] - self._recv_times[0]
                    if span > 0:
                        self.recv_fps = (len(self._recv_times) - 1) / span

    def latest(self) -> Optional[tuple[int, np.ndarray, float]]:
        with self._lock:
            return self._latest


# ---------- YOLOX ----------

def _yolox_letterbox(img: np.ndarray, target: int = 416):
    h, w = img.shape[:2]
    r = min(target / h, target / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target, target, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    return padded, r


class YoloxOnnx:
    """Person detector. mmdeploy-exported YOLOX with NMS baked into the graph.

    Input  : (1, 3, 416, 416) float32, BGR raw values (no normalization)
    Outputs: dets  (1, N, 5)  [x1, y1, x2, y2, score]
             labels (1, N)    int64 — class id (person == 0 here)
    """

    def __init__(self, model_path: str, providers, input_size: int = 416, score_thr: float = 0.5):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.score_thr = score_thr

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        padded, r = _yolox_letterbox(frame_bgr, self.input_size)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        dets, labels = self.session.run(None, {self.input_name: blob})
        dets = dets[0]  # (N, 5)
        labels = labels[0]
        if dets.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        mask = (labels == 0) & (dets[:, 4] >= self.score_thr)
        keep = dets[mask].astype(np.float32)
        if keep.size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        keep[:, :4] /= r
        return keep


# ---------- RTMPose ----------

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([pt[0] * c - pt[1] * s, pt[0] * s + pt[1] * c], dtype=np.float32)


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def _bbox_to_cs(bbox: np.ndarray, *, padding: float, aspect_w_over_h: float):
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1) * padding
    h = (y2 - y1) * padding
    if w > h * aspect_w_over_h:
        h = w / aspect_w_over_h
    else:
        w = h * aspect_w_over_h
    return np.array([cx, cy], dtype=np.float32), np.array([w, h], dtype=np.float32)


def _warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int, inv: bool = False) -> np.ndarray:
    src_w = scale[0]
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), 0.0)
    dst_dir = np.array([0.0, out_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _third_point(src[0], src[1])
    dst[0] = [out_w * 0.5, out_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _third_point(dst[0], dst[1])
    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


class RtmposeOnnx:
    """RTMPose SimCC.

    Input : (B, 3, 256, 192) float32, BGR, ImageNet-normalized
    Output: simcc_x (B, K, 384), simcc_y (B, K, 512)  (K = 17 for body7)
    """

    def __init__(self, model_path: str, providers, simcc_split_ratio: float = 2.0):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape  # [batch, 3, H, W]
        self.input_h = int(shape[2])
        self.input_w = int(shape[3])
        self.aspect = self.input_w / self.input_h
        self.simcc_split = simcc_split_ratio

    def _preprocess_one(self, frame_bgr: np.ndarray, bbox: np.ndarray):
        center, scale = _bbox_to_cs(bbox, padding=1.25, aspect_w_over_h=self.aspect)
        M = _warp_matrix(center, scale, self.input_w, self.input_h, inv=False)
        crop = cv2.warpAffine(
            frame_bgr, M, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR
        )
        crop = (crop.astype(np.float32) - IMAGENET_MEAN_BGR) / IMAGENET_STD_BGR
        blob = crop.transpose(2, 0, 1)
        M_inv = _warp_matrix(center, scale, self.input_w, self.input_h, inv=True)
        return blob, M_inv

    def infer(self, frame_bgr: np.ndarray, bboxes: np.ndarray):
        if bboxes.shape[0] == 0:
            return []
        blobs = []
        invs = []
        for bb in bboxes:
            blob, M_inv = self._preprocess_one(frame_bgr, bb)
            blobs.append(blob)
            invs.append(M_inv)
        batch = np.stack(blobs, axis=0)
        simcc_x, simcc_y = self.session.run(None, {self.input_name: batch})
        # argmax decode
        x_loc = np.argmax(simcc_x, axis=2).astype(np.float32)  # (B, K)
        y_loc = np.argmax(simcc_y, axis=2).astype(np.float32)
        max_x = np.max(simcc_x, axis=2)
        max_y = np.max(simcc_y, axis=2)
        scores = np.minimum(max_x, max_y)
        kpts_input = np.stack([x_loc, y_loc], axis=-1) / self.simcc_split  # in input frame
        results = []
        for i, M_inv in enumerate(invs):
            pts = kpts_input[i]
            # apply M_inv (2x3) to each (x, y)
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            homo = np.concatenate([pts, ones], axis=1)  # (K, 3)
            mapped = homo @ M_inv.T  # (K, 2)
            sc = scores[i].astype(np.float32)
            sc[sc <= 0] = 0.0
            results.append((mapped.astype(np.float32), sc))
        return results


# ---------- Drawer ----------

class PoseDrawer:
    def __init__(self, *, kp_thr: float = 0.3, line_thickness: int = 2):
        self.kp_thr = kp_thr
        self.lt = line_thickness

    def draw(self, frame_bgr: np.ndarray, persons, color):
        out = frame_bgr  # in-place edits
        for kpts, scores in persons:
            for a, b in COCO_SKELETON:
                if scores[a] >= self.kp_thr and scores[b] >= self.kp_thr:
                    p1 = (int(kpts[a, 0]), int(kpts[a, 1]))
                    p2 = (int(kpts[b, 0]), int(kpts[b, 1]))
                    cv2.line(out, p1, p2, color, self.lt, cv2.LINE_AA)
            for k in range(kpts.shape[0]):
                if scores[k] >= self.kp_thr:
                    cv2.circle(out, (int(kpts[k, 0]), int(kpts[k, 1])), 3, color, -1, cv2.LINE_AA)
        return out


# ---------- Engine ----------

@dataclasses.dataclass
class EngineStats:
    pose_count: int = 0
    pose_recent: deque = dataclasses.field(default_factory=lambda: deque(maxlen=60))
    avg_pose_fps: float = 0.0
    recent_pose_fps: float = 0.0
    last_stage_ms: float = 0.0
    started_at: float = dataclasses.field(default_factory=time.monotonic)


@dataclasses.dataclass
class PoseResult:
    seq: int
    captured_at: float
    processed_at: float
    persons: list  # list of (kpts (K,2), scores (K,))
    bboxes: np.ndarray


class PoseEngine:
    def __init__(
        self,
        det: YoloxOnnx,
        pose: RtmposeOnnx,
        *,
        det_frequency: int = 10,
        single_person: bool = True,
    ):
        self.det = det
        self.pose = pose
        self.det_frequency = max(1, int(det_frequency))
        self.single_person = single_person
        self._frame_idx = 0
        self._cached_bboxes: np.ndarray = np.zeros((0, 5), dtype=np.float32)

    def process(self, seq: int, frame_bgr: np.ndarray, captured_at: float) -> PoseResult:
        t0 = time.monotonic()
        do_detect = (self._frame_idx % self.det_frequency == 0) or self._cached_bboxes.shape[0] == 0
        if do_detect:
            dets = self.det.infer(frame_bgr)
            if self.single_person and dets.shape[0] > 1:
                areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
                dets = dets[np.argsort(-areas)[:1]]
            self._cached_bboxes = dets
        bboxes = self._cached_bboxes
        persons = self.pose.infer(frame_bgr, bboxes) if bboxes.shape[0] else []
        self._frame_idx += 1
        return PoseResult(
            seq=seq,
            captured_at=captured_at,
            processed_at=time.monotonic(),
            persons=persons,
            bboxes=bboxes,
        )


def update_stats(stats: EngineStats, result: PoseResult) -> None:
    now = result.processed_at
    stats.pose_count += 1
    stats.pose_recent.append(now)
    if len(stats.pose_recent) >= 2:
        span = stats.pose_recent[-1] - stats.pose_recent[0]
        if span > 0:
            stats.recent_pose_fps = (len(stats.pose_recent) - 1) / span
    elapsed = now - stats.started_at
    if elapsed > 0:
        stats.avg_pose_fps = stats.pose_count / elapsed
    stats.last_stage_ms = (result.processed_at - result.captured_at) * 1000.0


def warmup_session(session: ort.InferenceSession, n: int) -> None:
    if n <= 0:
        return
    inp = session.get_inputs()[0]
    shape = [int(d) if isinstance(d, int) else 1 for d in inp.shape]
    dummy = np.zeros(shape, dtype=np.float32)
    name = inp.name
    for _ in range(n):
        session.run(None, {name: dummy})


# ---------- helpers for CLI scripts ----------

def build_engines_for(camera_count: int, args) -> list[PoseEngine]:
    """Construct one engine per camera, sharing model bytes by re-loading per session.

    We keep one session per camera so that capture threads don't contend on the
    GIL inside the runtime's Python wrapper.
    """
    det_providers = select_providers(
        args.device,
        trt_cache_dir=getattr(args, "trt_cache_dir", None),
        trt_fp16=getattr(args, "trt_fp16", False),
        use_trt=(args.device == "tensorrt"
                 and getattr(args, "trt_models", "det") in ("det", "all")),
    )
    pose_providers = select_providers(
        args.device,
        trt_cache_dir=getattr(args, "trt_cache_dir", None),
        trt_fp16=getattr(args, "trt_fp16", False),
        use_trt=(args.device == "tensorrt"
                 and getattr(args, "trt_models", "det") in ("pose", "all")),
    )

    engines: list[PoseEngine] = []
    for _ in range(camera_count):
        det = YoloxOnnx(args.det_model, det_providers, score_thr=args.det_score)
        pose = RtmposeOnnx(args.pose_model, pose_providers)
        warmup_session(det.session, getattr(args, "trt_warmup_frames", 0))
        warmup_session(pose.session, getattr(args, "trt_warmup_frames", 0))
        engines.append(
            PoseEngine(
                det,
                pose,
                det_frequency=args.det_frequency,
                single_person=not args.multi_person,
            )
        )
    return engines


def add_common_args(parser) -> None:
    """argparse arguments shared by all CLI tools."""
    parser.add_argument("--cam0", default=DEFAULT_CAM0)
    parser.add_argument("--cam1", default=DEFAULT_CAM1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "tensorrt"],
    )
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    parser.add_argument("--det-score", type=float, default=0.5)
    parser.add_argument("--det-frequency", type=int, default=10)
    parser.add_argument("--multi-person", action="store_true")
    parser.add_argument("--kp-thr", type=float, default=0.3)
    parser.add_argument("--log-every", type=float, default=10.0,
                        help="seconds between stats lines")
    parser.add_argument("--trt-cache-dir", default="outputs/tensorrt_engines")
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument("--trt-warmup-frames", type=int, default=0)
    parser.add_argument("--trt-models", default="det",
                        choices=["det", "pose", "all"])
