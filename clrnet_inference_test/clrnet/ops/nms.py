"""Compatibility shim for official CLRNet's ``clrnet.ops.nms`` import path."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clrnet_common.nms_loader import load_nms_impl


nms_impl = load_nms_impl()


def nms(boxes, scores, overlap, top_k):
    return nms_impl.nms_forward(boxes, scores, overlap, top_k)
