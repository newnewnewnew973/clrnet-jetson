"""Load the local external CUDA NMS extension for inference tests."""

import importlib.util
import sys
from pathlib import Path


def _load_nms_impl():
    ext_dir = Path(__file__).resolve().parents[2] / "extensions" / "nms"
    candidates = sorted(ext_dir.glob("nms_impl*.so"))
    if not candidates:
        raise ImportError(
            "external nms_impl is not built. Build it with:\n"
            "  cd /home/newnew/workspace/clrnet_inference_test/extensions/nms\n"
            "  python setup.py build_ext --inplace"
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("nms_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load nms extension: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("nms_impl", module)
    spec.loader.exec_module(module)
    return module


nms_impl = _load_nms_impl()


def nms(boxes, scores, overlap, top_k):
    return nms_impl.nms_forward(boxes, scores, overlap, top_k)
