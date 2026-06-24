"""Load the workspace-local CLRNet CUDA NMS extension."""

import importlib.util
import sys
from types import ModuleType
from pathlib import Path

from .runtime import PROJECT_ROOT


DEFAULT_NMS_EXTENSION_DIR = PROJECT_ROOT / "clrnet_common/extensions/nms"


def load_nms_impl(extension_dir: Path = DEFAULT_NMS_EXTENSION_DIR) -> ModuleType:
    """Load the compiled nms_impl extension from the given extension directory."""
    cached_module = sys.modules.get("nms_impl")
    if cached_module is not None:
        return cached_module

    candidates = sorted(extension_dir.glob("nms_impl*.so"))
    if not candidates:
        raise ImportError(
            "external nms_impl is not built. Build it with:\n"
            f"  cd {extension_dir}\n"
            "  python setup.py build_ext --inplace"
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("nms_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load nms extension: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["nms_impl"] = module
    spec.loader.exec_module(module)
    return module
