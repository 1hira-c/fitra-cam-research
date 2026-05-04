#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import site
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
user_site = site.getusersitepackages()
sys.path = [path for path in sys.path if path != user_site]

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dual_rtmpose_core import (
    DualCameraPoseRunner,
    PoseSnapshotStore,
    add_common_camera_args,
    build_runner_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual-camera RTMPose and stream 2D skeletons to a Web app."
    )
    add_common_camera_args(parser, include_display=False)
    parser.set_defaults(save_every=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
    )
    return parser.parse_args()


def create_app(runner: DualCameraPoseRunner, store: PoseSnapshotStore) -> FastAPI:
    static_dir = Path(__file__).resolve().parent.parent / "web" / "dual_rtmpose"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runner.start_background(store)
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(title="fitra-cam dual RTMPose Web UI", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.websocket("/ws/poses")
    async def poses_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        last_bundle_sequence = -1
        try:
            while True:
                bundle = store.get_bundle()
                if bundle["bundle_sequence"] != last_bundle_sequence:
                    await websocket.send_json(bundle)
                    last_bundle_sequence = bundle["bundle_sequence"]
                await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            return

    return app


def main() -> int:
    args = parse_args()
    config = build_runner_config(args, allow_display=False)
    store = PoseSnapshotStore()
    runner = DualCameraPoseRunner(config, publisher=store.publish_camera_payload)
    app = create_app(runner, store)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
