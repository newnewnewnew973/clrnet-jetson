#!/usr/bin/env python3
"""Verify which CLRNet modules come from local overrides or official source."""

import importlib
import sys
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[2]
LOCAL_PROJECT_DIR = WORKSPACE_DIR / "clrnet_inference_test"
OFFICIAL_CLRNET_DIR = WORKSPACE_DIR / "clrnet"
OFFICIAL_PACKAGE_DIR = OFFICIAL_CLRNET_DIR / "clrnet"


def reset_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "clrnet"
            or name.startswith("clrnet.")
            or name == "mmcv"
            or name.startswith("mmcv.")
            or name == "torchvision"
            or name.startswith("torchvision.")
        ):
            sys.modules.pop(name, None)


def module_file(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    file = getattr(module, "__file__", None)
    if file is None:
        raise AssertionError(f"{module_name} has no __file__")
    return Path(file).resolve()


def assert_under(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root.resolve()):
        raise AssertionError(f"{label}: {path} is not under {root.resolve()}")


def main() -> int:
    reset_modules()

    # Local project must precede the official repo so test-only overrides
    # such as clrnet.ops.nms, mmcv, and torchvision win import resolution.
    # The proxy clrnet package then falls back to official CLRNet for modules
    # that are not overridden locally, e.g. clrnet.models and clrnet.datasets.
    sys.path.insert(0, str(OFFICIAL_CLRNET_DIR))
    sys.path.insert(0, str(LOCAL_PROJECT_DIR))

    # Each entry is (module_name, expected_root). Import the module and assert
    # module.__file__ is under the intended source tree.
    checks = [
        ("clrnet", LOCAL_PROJECT_DIR),
        ("clrnet.ops", LOCAL_PROJECT_DIR),
        ("clrnet.ops.nms", LOCAL_PROJECT_DIR),
        ("clrnet.models", OFFICIAL_PACKAGE_DIR),
        ("clrnet.datasets", OFFICIAL_PACKAGE_DIR),
        ("clrnet.utils.config", OFFICIAL_PACKAGE_DIR),
        ("mmcv", LOCAL_PROJECT_DIR),
        ("mmcv.cnn", LOCAL_PROJECT_DIR),
        ("mmcv.parallel", LOCAL_PROJECT_DIR),
        ("mmcv.runner", LOCAL_PROJECT_DIR),
        ("torchvision", LOCAL_PROJECT_DIR),
    ]

    for module_name, expected_root in checks:
        path = module_file(module_name)
        assert_under(path, expected_root, module_name)
        print(f"[OK] {module_name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
