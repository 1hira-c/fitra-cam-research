#!/usr/bin/env python3
import argparse
import os
import site
import sys

os.environ.setdefault("PYTHONNOUSERSITE", "1")
user_site = site.getusersitepackages()
sys.path = [path for path in sys.path if path != user_site]

from dual_rtmpose_core import (
    DualCameraPoseRunner,
    add_common_camera_args,
    build_runner_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual-camera 2D pose estimation with RTMPose."
    )
    add_common_camera_args(parser, include_display=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_runner_config(args, allow_display=True)
    runner = DualCameraPoseRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
