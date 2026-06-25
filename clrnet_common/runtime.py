"""Runtime helpers shared by CLRNet evaluation, latency, and conversion tools."""

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PROJECT_ROOT = PROJECT_ROOT / "clrnet_inference"
OFFICIAL_CLRNET_ROOT = PROJECT_ROOT / "clrnet"

# These buffers/weights are not required for inference in exported CULane
# checkpoints. Any other checkpoint mismatch is treated as unsafe.
EXPECTED_MISSING_KEYS = {
    "heads.criterion.weight",
    "heads.prior_feat_ys",
    "heads.prior_ys",
    "heads.sample_x_indexs",
}


def ensure_numpy_bool_alias() -> None:
    """Restore the deprecated np.bool alias expected by older CLRNet code."""
    if "bool" not in np.__dict__:
        np.bool = bool


def configure_import_paths(local_project_root: Path | None = None) -> None:
    """Make local proxy modules and the official CLRNet checkout importable."""
    proxy_root = local_project_root or INFERENCE_PROJECT_ROOT
    if str(OFFICIAL_CLRNET_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_CLRNET_ROOT))
    if str(proxy_root) not in sys.path:
        sys.path.insert(0, str(proxy_root))


def nms_build_message(include_arch: bool = True) -> str:
    """Return the build hint shown when the external CUDA NMS module is missing."""
    command = "  TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace"
    if not include_arch:
        command = "  python setup.py build_ext --inplace"
    return (
        "External CLRNet CUDA NMS extension is not built.\n"
        "Build it first:\n"
        "  cd clrnet_common/extensions/nms\n"
        f"{command}\n"
    )


def default_device_arg() -> str:
    """Prefer CUDA when available, while keeping CPU smoke tests usable."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_culane_model_args(
    model: str,
    config_path: str | None,
    checkpoint_path: str | None,
) -> tuple[str, str, str]:
    """Resolve a CULane model preset into config and checkpoint paths."""
    model_name = model
    config = config_path or str(
        OFFICIAL_CLRNET_ROOT / f"configs/clrnet/clr_{model_name}_culane.py"
    )
    checkpoint = checkpoint_path or str(PROJECT_ROOT / f"weights/culane_{model_name}.pth")
    return model_name, config, checkpoint


def load_checkpoint_for_inference(model, checkpoint_path: Path):
    """Load an inference checkpoint and fail on unexpected state-dict drift."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = (
        checkpoint["net"]
        if isinstance(checkpoint, dict) and "net" in checkpoint
        else checkpoint
    )
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
    """Validate the requested torch device before starting a long run."""
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check Jetson GPU access, CUDA/PyTorch install, and sandbox/device permissions."
        )
    return device
