"""Shared runtime helpers for CLRNet inference/evaluation scripts."""

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROJECT_ROOT = PROJECT_ROOT / "clrnet_inference_test"
OFFICIAL_CLRNET_ROOT = PROJECT_ROOT / "clrnet"
EXPECTED_MISSING_KEYS = {
    "heads.criterion.weight",
    "heads.prior_feat_ys",
    "heads.prior_ys",
    "heads.sample_x_indexs",
}


def ensure_numpy_bool_alias() -> None:
    if "bool" not in np.__dict__:
        np.bool = bool


def configure_import_paths() -> None:
    if str(OFFICIAL_CLRNET_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_CLRNET_ROOT))
    if str(LOCAL_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(LOCAL_PROJECT_ROOT))


def nms_build_message(include_arch: bool = True) -> str:
    command = "  TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace"
    if not include_arch:
        command = "  python setup.py build_ext --inplace"
    return (
        "External CLRNet CUDA NMS extension is not built.\n"
        "Build it first:\n"
        "  cd /home/newnew/workspace/clrnet_inference_test/extensions/nms\n"
        f"{command}\n"
    )


def default_device_arg() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint_for_inference(model, checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
    state = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    loaded = len(set(model.state_dict().keys()) & set(state.keys()))
    print(f"checkpoint_loaded_keys={loaded}/{len(model.state_dict())}")
    if incompatible.missing_keys:
        print(f"checkpoint_missing_keys={len(incompatible.missing_keys)}")
        for key in incompatible.missing_keys:
            print(f"  missing: {key}")
    if incompatible.unexpected_keys:
        print(f"checkpoint_unexpected_keys={len(incompatible.unexpected_keys)}")
        for key in incompatible.unexpected_keys:
            print(f"  unexpected: {key}")

    unsafe_missing = set(incompatible.missing_keys) - EXPECTED_MISSING_KEYS
    if unsafe_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match this CLRNet model. "
            f"unsafe_missing={sorted(unsafe_missing)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    return incompatible


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check Jetson GPU access, CUDA/PyTorch install, and sandbox/device permissions."
        )
    return device
