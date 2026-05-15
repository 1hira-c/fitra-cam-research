#!/usr/bin/env python3
"""Compare two keypoint JSONL dumps (one from Python ORT, one from C++ TRT)
and print per-frame and aggregate statistics.

Pass/fail criteria from docs/cpp-migration-plan.md Phase 1:
  - per-person bbox IoU > 0.99
  - per-keypoint L2 distance < 1.0 px

Each line in the JSONL files must have the shape:
  {"frame": N, "persons":
    [{"bbox": [x1,y1,x2,y2,score],
      "kpts": [[x,y,score], ...17]}]}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True, help="Python ORT JSONL")
    ap.add_argument("--candidate", required=True, help="C++ TRT JSONL")
    ap.add_argument("--iou-threshold", type=float, default=0.99,
                    help="minimum bbox IoU per person (default 0.99)")
    ap.add_argument("--kpt-threshold", type=float, default=1.0,
                    help="maximum keypoint L2 distance in pixels (default 1.0)")
    ap.add_argument("--score-threshold", type=float, default=0.3,
                    help="ignore keypoints with reference score below this (default 0.3)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print per-frame diffs")
    args = ap.parse_args()

    ref = load_jsonl(Path(args.reference))
    cand = load_jsonl(Path(args.candidate))
    if len(ref) != len(cand):
        print(f"[warn] frame count differs: ref={len(ref)} cand={len(cand)}",
              file=sys.stderr)
    n_frames = min(len(ref), len(cand))

    bbox_ious: list[float] = []
    kpt_dists: list[float] = []
    score_diffs: list[float] = []
    per_frame_max_kpt: list[float] = []
    skipped_frames = 0

    for i in range(n_frames):
        r = ref[i]
        c = cand[i]
        if len(r["persons"]) != len(c["persons"]):
            skipped_frames += 1
            if args.verbose:
                print(f"frame {i}: person count differs ref={len(r['persons'])} "
                      f"cand={len(c['persons'])}")
            continue
        frame_max_kpt = 0.0
        for pr, pc in zip(r["persons"], c["persons"]):
            iou = iou_xyxy(pr["bbox"], pc["bbox"])
            bbox_ious.append(iou)
            for kr, kc in zip(pr["kpts"], pc["kpts"]):
                # only compare keypoints the reference considers reliable
                if kr[2] < args.score_threshold:
                    continue
                dx = kr[0] - kc[0]
                dy = kr[1] - kc[1]
                d = math.hypot(dx, dy)
                kpt_dists.append(d)
                score_diffs.append(abs(kr[2] - kc[2]))
                if d > frame_max_kpt:
                    frame_max_kpt = d
        per_frame_max_kpt.append(frame_max_kpt)
        if args.verbose:
            print(f"frame {i}: max_kpt={frame_max_kpt:.3f} px")

    def fmt_stats(name: str, values: list[float], unit: str) -> str:
        if not values:
            return f"{name}: <no samples>"
        return (
            f"{name}: n={len(values)} "
            f"min={min(values):.4f} "
            f"mean={statistics.fmean(values):.4f} "
            f"p50={percentile(values, 50):.4f} "
            f"p95={percentile(values, 95):.4f} "
            f"p99={percentile(values, 99):.4f} "
            f"max={max(values):.4f} {unit}"
        )

    print("=" * 60)
    print(f"frames compared : {n_frames}  skipped(person count mismatch): {skipped_frames}")
    print(fmt_stats("bbox IoU       ", bbox_ious, ""))
    print(fmt_stats("kpt L2         ", kpt_dists, "px"))
    print(fmt_stats("kpt score diff ", score_diffs, ""))
    print(fmt_stats("frame max kpt L2", per_frame_max_kpt, "px"))
    print("=" * 60)

    ok = True
    if bbox_ious and min(bbox_ious) < args.iou_threshold:
        print(f"FAIL: min bbox IoU {min(bbox_ious):.4f} < {args.iou_threshold}")
        ok = False
    if kpt_dists and max(kpt_dists) > args.kpt_threshold:
        print(f"FAIL: max kpt L2 {max(kpt_dists):.4f}px > {args.kpt_threshold}px")
        ok = False
    if skipped_frames > 0:
        print(f"FAIL: {skipped_frames} frames had differing person count")
        ok = False
    if ok:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
