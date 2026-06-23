import argparse
import json
import statistics
import sys
import time
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
        description="Measure PyTorch CLRNet DLA34 CULane inference latency/FPS."
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
        default=str(PROJECT_ROOT / "clrnet_inference_test/outputs/latency/pytorch_dla34"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check Jetson GPU access, CUDA/PyTorch install, and sandbox/device permissions."
        )
    return device


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def build_culane_dataset(cfg, data_root: Path):
    cfg.dataset.test.data_root = str(data_root)
    return build_dataset(cfg.dataset.test, cfg)


def summarize_ms(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def write_report(path: Path, result: dict):
    lines = [
        "# PyTorch CLRNet DLA34 Latency Baseline",
        "",
        "This latency measurement uses the official CLRNet CULane dataset class and the",
        "local external CUDA NMS extension.",
        "",
        "## Summary",
        "",
        f"- Device: `{result['device']}`",
        f"- Samples: `{result['samples']}`",
        f"- Warmup: `{result['warmup']}`",
        f"- End-to-end FPS: `{result['fps_e2e']:.2f}`",
        f"- End-to-end mean latency: `{result['latency_ms']['e2e']['mean']:.2f} ms`",
        f"- Dataset/preprocess mean latency: `{result['latency_ms']['preprocess']['mean']:.2f} ms`",
        f"- Model forward mean latency: `{result['latency_ms']['forward']['mean']:.2f} ms`",
        f"- Postprocess mean latency: `{result['latency_ms']['postprocess']['mean']:.2f} ms`",
        "",
        "## Notes",
        "",
        "- Timing excludes image/result file writing.",
        "- CUDA timings use `torch.cuda.synchronize()` around measured regions.",
        "- Postprocess includes `model.heads.get_lanes()`, including CLRNet NMS.",
        "- This is the PyTorch latency/FPS baseline used before ONNX/TensorRT comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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

    device = resolve_device(args.device)
    dataset = build_culane_dataset(cfg, data_root)
    total = min(args.limit, len(dataset))
    warmup = min(args.warmup, total)

    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint)
    model.eval()

    print(f"data_root={dataset.data_root}")
    print(f"dataset_total={len(dataset)}")
    print(f"measured_samples={total}")
    print(f"warmup={warmup}")
    print(f"device={device}")
    print(f"output_dir={output_dir}")

    with torch.no_grad():
        for idx in range(warmup):
            sample = dataset[idx]
            tensor = sample["img"].unsqueeze(0).contiguous().to(device)
            output = model(tensor)
            _ = model.heads.get_lanes(output)[0]
        synchronize(device)

        preprocess_ms = []
        forward_ms = []
        postprocess_ms = []
        e2e_ms = []
        lane_counts = []

        for idx in tqdm(range(total), desc="pytorch-latency-measurement"):
            start_e2e = time.perf_counter()

            start = time.perf_counter()
            sample = dataset[idx]
            tensor = sample["img"].unsqueeze(0).contiguous().to(device)
            synchronize(device)
            preprocess_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            output = model(tensor)
            synchronize(device)
            forward_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            lanes = model.heads.get_lanes(output)[0]
            synchronize(device)
            postprocess_ms.append((time.perf_counter() - start) * 1000.0)
            lane_counts.append(len(lanes))

            e2e_ms.append((time.perf_counter() - start_e2e) * 1000.0)

    result = {
        "device": str(device),
        "samples": total,
        "warmup": warmup,
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "fps_e2e": 1000.0 / statistics.fmean(e2e_ms),
        "latency_ms": {
            "preprocess": summarize_ms(preprocess_ms),
            "forward": summarize_ms(forward_ms),
            "postprocess": summarize_ms(postprocess_ms),
            "e2e": summarize_ms(e2e_ms),
        },
        "lane_count": {
            "mean": statistics.fmean(lane_counts),
            "min": min(lane_counts),
            "max": max(lane_counts),
        },
    }

    json_path = output_dir / "pytorch_latency_dla34.json"
    report_path = output_dir / "PYTORCH_LATENCY_BASELINE.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(report_path, result)

    print(f"fps_e2e={result['fps_e2e']:.2f}")
    print(f"latency_e2e_mean_ms={result['latency_ms']['e2e']['mean']:.2f}")
    print(f"latency_forward_mean_ms={result['latency_ms']['forward']['mean']:.2f}")
    print(f"latency_json={json_path}")
    print(f"latency_report={report_path}")


if __name__ == "__main__":
    main()
