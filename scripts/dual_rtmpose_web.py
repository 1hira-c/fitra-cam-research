#!/usr/bin/env python3
"""FastAPI + WebSocket skeleton viewer for two USB cameras.

The HTTP server runs in the main thread, two capture+pose worker threads
write the latest result into a shared snapshot, and an asyncio task
publishes that snapshot to every connected WebSocket client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WEB_DIR = REPO_ROOT / "web" / "dual_rtmpose"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # noqa: E402
    CameraConfig,
    CameraReader,
    EngineStats,
    add_common_args,
    build_engines_for,
    update_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual USB cam RTMPose WebSocket server")
    add_common_args(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--publish-every", type=int, default=1,
                        help="publish every N-th processed frame per camera (1 = every frame)")
    return parser.parse_args()


class CameraRuntime:
    def __init__(self, cam_id: int, cfg: CameraConfig, engine, *, publish_every: int):
        self.cam_id = cam_id
        self.cfg = cfg
        self.engine = engine
        self.publish_every = max(1, int(publish_every))
        self.stats = EngineStats()
        self.reader = CameraReader(cam_id, cfg)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_bundle: dict = self._empty_bundle()
        self._lock = threading.Lock()
        self._published_seq = 0
        self._processed_count = 0

    def _empty_bundle(self) -> dict:
        return {
            "id": self.cam_id,
            "w": self.cfg.width,
            "h": self.cfg.height,
            "persons": [],
            "stats": {
                "recv_fps": 0.0,
                "recent_pose_fps": 0.0,
                "avg_pose_fps": 0.0,
                "pending": 0,
                "stage_ms": 0.0,
                "processed": 0,
            },
        }

    def start(self) -> None:
        self.reader.start()
        self._thread = threading.Thread(target=self._loop, name=f"PoseRuntime-{self.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.reader.stop()

    def _loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            latest = self.reader.latest()
            if latest is None or latest[0] == last_seq:
                time.sleep(0.002)
                continue
            seq, frame, captured_at = latest
            last_seq = seq
            result = self.engine.process(seq, frame, captured_at)
            update_stats(self.stats, result)
            self._processed_count += 1
            if self._processed_count % self.publish_every != 0:
                continue
            persons_json = []
            for kpts, scores in result.persons:
                persons_json.append({
                    "kpts": [[float(p[0]), float(p[1]), float(s)] for p, s in zip(kpts, scores)],
                })
            for i, bb in enumerate(result.bboxes):
                if i < len(persons_json):
                    persons_json[i]["bbox"] = [float(v) for v in bb.tolist()]
            pending = max(self.reader.seq - self.stats.pose_count, 0)
            bundle = {
                "id": self.cam_id,
                "w": self.cfg.width,
                "h": self.cfg.height,
                "persons": persons_json,
                "stats": {
                    "recv_fps": round(self.reader.recv_fps, 2),
                    "recent_pose_fps": round(self.stats.recent_pose_fps, 2),
                    "avg_pose_fps": round(self.stats.avg_pose_fps, 2),
                    "pending": pending,
                    "stage_ms": round(self.stats.last_stage_ms, 1),
                    "processed": self.stats.pose_count,
                    "captured_at_ms": int(captured_at * 1000),
                },
            }
            with self._lock:
                self._latest_bundle = bundle
                self._published_seq = seq

    def snapshot(self) -> tuple[dict, int]:
        with self._lock:
            return self._latest_bundle, self._published_seq


class WebState:
    def __init__(self, runtimes: list[CameraRuntime]):
        self.runtimes = runtimes
        self.clients: set[WebSocket] = set()
        self.clients_lock = asyncio.Lock()
        self._seq = 0

    def make_bundle(self) -> dict:
        self._seq += 1
        cams = [rt.snapshot()[0] for rt in self.runtimes]
        return {
            "seq": self._seq,
            "ts_ms": int(time.time() * 1000),
            "cameras": cams,
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    runtimes: list[CameraRuntime] = []
    web_state: Optional[WebState] = None
    publisher_task: Optional[asyncio.Task] = None
    log_task: Optional[asyncio.Task] = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal web_state, publisher_task, log_task
        cam_cfgs = [
            CameraConfig(path=args.cam0, width=args.width, height=args.height,
                         fps=args.fps, fourcc=args.fourcc),
            CameraConfig(path=args.cam1, width=args.width, height=args.height,
                         fps=args.fps, fourcc=args.fourcc),
        ]
        print(f"[setup] opening cameras: {args.cam0} / {args.cam1}", file=sys.stderr)
        print(f"[setup] device={args.device}, building engines...", file=sys.stderr)
        engines = build_engines_for(len(cam_cfgs), args)
        for i, (cfg, engine) in enumerate(zip(cam_cfgs, engines)):
            rt = CameraRuntime(i, cfg, engine, publish_every=args.publish_every)
            rt.start()
            runtimes.append(rt)
        web_state = WebState(runtimes)
        publisher_task = asyncio.create_task(_publisher_loop(web_state))
        log_task = asyncio.create_task(_logger_loop(runtimes, args.log_every))
        print(f"[ready] http://{args.host}:{args.port}/ (Ctrl-C to stop)", file=sys.stderr)
        try:
            yield
        finally:
            print("[shutdown] stopping workers...", file=sys.stderr)
            if publisher_task:
                publisher_task.cancel()
            if log_task:
                log_task.cancel()
            for rt in runtimes:
                rt.stop()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        assert web_state is not None
        async with web_state.clients_lock:
            web_state.clients.add(ws)
        try:
            while True:
                # passively receive (clients may send ping); we publish via task
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            async with web_state.clients_lock:
                web_state.clients.discard(ws)

    @app.get("/stats")
    async def stats():
        if web_state is None:
            return {"ready": False}
        return web_state.make_bundle()

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")

    async def _publisher_loop(state: WebState):
        publish_period = 1.0 / 30.0  # publish at most 30Hz independently of pose fps
        while True:
            await asyncio.sleep(publish_period)
            bundle = state.make_bundle()
            msg = json.dumps(bundle, separators=(",", ":"))
            dead: list[WebSocket] = []
            async with state.clients_lock:
                clients = list(state.clients)
            for ws in clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            if dead:
                async with state.clients_lock:
                    for ws in dead:
                        state.clients.discard(ws)

    async def _logger_loop(rts: list[CameraRuntime], interval: float):
        while True:
            await asyncio.sleep(interval)
            for rt in rts:
                pending = max(rt.reader.seq - rt.stats.pose_count, 0)
                print(
                    f"[stats] cam{rt.cam_id}: recv={rt.reader.recv_fps:5.2f} "
                    f"avg_pose={rt.stats.avg_pose_fps:5.2f} recent_pose={rt.stats.recent_pose_fps:5.2f} "
                    f"stage_ms={rt.stats.last_stage_ms:6.1f} processed={rt.stats.pose_count} pending={pending}",
                    file=sys.stderr,
                )

    return app


def main() -> int:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
