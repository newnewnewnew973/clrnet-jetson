#!/usr/bin/env python3
"""Run functional tests for the external CLRNet CUDA NMS extension."""

import sys
from pathlib import Path

import torch


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
LOCAL_PROJECT_DIR = WORKSPACE_DIR / "clrnet_inference_test"
OFFICIAL_CLRNET_DIR = WORKSPACE_DIR / "clrnet"
PROP_SIZE = 77


def make_lane(offset_value: float) -> torch.Tensor:
    lane = torch.zeros(PROP_SIZE, dtype=torch.float32)
    lane[2] = 0.0
    lane[4] = 72.0
    lane[5:] = offset_value
    return lane


def run_nms_case(nms, name: str, offsets, scores, overlap, top_k, expected_keep) -> bool:
    boxes = torch.stack([make_lane(offset) for offset in offsets]).cuda()
    score_tensor = torch.tensor(scores, dtype=torch.float32, device="cuda")

    keep, num_to_keep, parent_object_index = nms(boxes, score_tensor, overlap, top_k)
    kept = keep[: int(num_to_keep.item())].cpu().tolist()

    if kept != expected_keep:
        print(f"[FAIL] {name}")
        print(f"offsets:       {offsets}")
        print(f"scores:        {scores}")
        print(f"overlap:       {overlap}")
        print(f"top_k:         {top_k}")
        print(f"expected keep: {expected_keep}")
        print(f"actual keep:   {kept}")
        print(f"parent index:  {parent_object_index.cpu().tolist()}")
        return False

    print(f"[OK] {name}: keep={kept}")
    return True


def main() -> int:
    sys.path.insert(0, str(OFFICIAL_CLRNET_DIR))
    sys.path.insert(0, str(LOCAL_PROJECT_DIR))

    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available")
        return 1

    try:
        from clrnet.ops.nms import nms
    except ImportError as exc:
        print("[FAIL] import nms")
        print(f"error: {exc}")
        return 1

    print("[OK] import nms")

    cases = [
        [
            "duplicate suppression",
            [10.0, 10.0, 40.0],
            [0.9, 0.8, 0.7],
            1.0,
            10,
            [0, 2],
        ],
        [
            "no duplicate",
            [10.0, 20.0, 40.0],
            [0.9, 0.8, 0.7],
            1.0,
            10,
            [0, 1, 2],
        ],
        [
            "score ordering",
            [10.0, 10.0, 40.0],
            [0.2, 0.9, 0.7],
            1.0,
            10,
            [1, 2],
        ],
        [
            "top_k limit",
            [10.0, 20.0, 40.0],
            [0.9, 0.8, 0.7],
            1.0,
            1,
            [0],
        ],
        [
            "threshold boundary",
            [10.0, 10.5, 11.0],
            [0.9, 0.8, 0.7],
            1.0,
            10,
            [0, 2],
        ],
    ]

    for case in cases:
        if not run_nms_case(nms, *case):
            return 1

    torch.cuda.synchronize()
    print("[OK] all nms tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
