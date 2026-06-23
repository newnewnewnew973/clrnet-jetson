import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

if "bool" not in np.__dict__:
    np.bool = bool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PROJECT_ROOT = PROJECT_ROOT / "clrnet_inference_test"
OFFICIAL_CLRNET_ROOT = PROJECT_ROOT / "clrnet"
EXPECTED_MISSING_KEYS = {
    "heads.criterion.weight",
    "heads.prior_feat_ys",
    "heads.prior_ys",
    "heads.sample_x_indexs",
}
if str(OFFICIAL_CLRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CLRNET_ROOT))
if str(LOCAL_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_ROOT))

try:
    import clrnet.datasets  # noqa: F401,E402 - registers official datasets/processes
    import clrnet.models  # noqa: F401,E402 - registers official model modules
    from clrnet.datasets.registry import build_dataset  # noqa: E402
    from clrnet.models.registry import build_net  # noqa: E402
    from clrnet.utils.config import Config  # noqa: E402
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(
            "External CLRNet CUDA NMS extension is not built.\n"
            "Build it first:\n"
            "  cd /home/newnew/workspace/clrnet_inference_test/extensions/nms\n"
            "  TORCH_CUDA_ARCH_LIST=8.7 python setup.py build_ext --inplace\n"
        ) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLRNet DLA34 on CULane using the official dataset class."
    )
    parser.add_argument(
        "--config",
        default=str(OFFICIAL_CLRNET_ROOT / "configs/clrnet/clr_dla34_culane.py"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights/culane_dla34.pth"),
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "clrnet_inference_test/data/CULane"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "clrnet_inference_test/outputs/eval/dla34_official"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check Jetson GPU access, CUDA/PyTorch install, and sandbox/device permissions."
        )
    return device


def build_culane_dataset(cfg, data_root: Path):
    cfg.dataset.test.data_root = str(data_root)
    dataset = build_dataset(cfg.dataset.test, cfg)
    return dataset


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False

    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_culane_dataset(cfg, data_root)
    total = len(dataset)

    device = resolve_device(args.device)
    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint)
    model.eval()

    print(f"data_root={dataset.data_root}")
    print(f"dataset_class={dataset.__class__.__name__}")
    print(f"dataset_total={total}")
    print(f"eval_list={dataset.list_path}")
    print(f"output_dir={output_dir}")

    predictions = []
    with torch.no_grad():
        for idx in tqdm(range(total), desc="official-dataset-eval"):
            sample = dataset[idx]
            tensor = sample["img"].unsqueeze(0).contiguous().to(device)
            output = model(tensor)
            lanes = model.heads.get_lanes(output)[0]
            predictions.append(lanes)

    if device.type == "cuda":
        torch.cuda.synchronize()

    f1 = dataset.evaluate(predictions, str(output_dir))
    print(f"official_dataset_evaluate_F1={f1:.6f}")


if __name__ == "__main__":
    main()
