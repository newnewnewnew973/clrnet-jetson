#!/usr/bin/env python3
"""Check whether the external CLRNet CUDA NMS extension can be imported."""

import sys
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
INFERENCE_PROJECT_DIR = WORKSPACE_DIR / "clrnet_inference"
OFFICIAL_CLRNET_DIR = WORKSPACE_DIR / "clrnet"


def main() -> int:
    sys.path.insert(0, str(OFFICIAL_CLRNET_DIR))
    sys.path.insert(0, str(INFERENCE_PROJECT_DIR))

    try:
        from clrnet.ops.nms import nms
    except ImportError as exc:
        print("[FAIL] from clrnet.ops.nms import nms")
        print(f"error: {exc}")
        return 1

    print("[OK] from clrnet.ops.nms import nms")
    print(f"nms: {nms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
