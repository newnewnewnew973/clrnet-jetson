import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

from runtime import (
    OFFICIAL_CLRNET_ROOT,
    PROJECT_ROOT,
    configure_import_paths,
    default_device_arg,
    ensure_numpy_bool_alias,
    load_checkpoint_for_inference,
    nms_build_message,
    resolve_device,
)


ensure_numpy_bool_alias()
configure_import_paths()

try:
    import clrnet.datasets  # noqa: F401,E402 - registers official datasets/processes
    import clrnet.models  # noqa: F401,E402 - registers official model modules
    from clrnet.datasets.registry import build_dataset  # noqa: E402
    from clrnet.models.registry import build_net  # noqa: E402
    from clrnet.utils.config import Config  # noqa: E402
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(nms_build_message()) from exc
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


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
    if args.device is None:
        args.device = default_device_arg()
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
