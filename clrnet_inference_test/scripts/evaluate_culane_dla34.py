import argparse
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
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--run-official-evaluate",
        action="store_true",
        help=(
            "Also run the official dataset.evaluate() after writing predictions. "
            "This uses Pool(cpu_count()) inside CLRNet and is not recommended on Jetson."
        ),
    )
    return parser.parse_args()


def build_culane_dataset(cfg, data_root: Path):
    cfg.dataset.test.data_root = str(data_root)
    dataset = build_dataset(cfg.dataset.test, cfg)
    return dataset


def write_culane_predictions(dataset, predictions, output_dir: Path) -> None:
    print("Generating prediction output...")
    for idx, pred in enumerate(predictions):
        relative_img = Path(dataset.data_infos[idx]["img_name"])
        pred_dir = output_dir / relative_img.parent
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"{relative_img.stem}.lines.txt"
        pred_file.write_text(dataset.get_prediction_string(pred), encoding="utf-8")


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

    write_culane_predictions(dataset, predictions, output_dir)
    print(f"predictions_written={total}")

    if args.run_official_evaluate:
        f1 = dataset.evaluate(predictions, str(output_dir))
        print(f"official_dataset_evaluate_F1={f1:.6f}")
    else:
        print(
            "official_dataset_evaluate_skipped=True "
            "(use scripts/measure_culane_metric.py --workers 4 for Jetson metric)"
        )


if __name__ == "__main__":
    main()
