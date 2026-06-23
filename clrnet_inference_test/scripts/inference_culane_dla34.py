import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

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
    import clrnet.models  # noqa: F401,E402 - registers official model modules
    from clrnet.models.registry import build_net  # noqa: E402
    from clrnet.utils.config import Config  # noqa: E402
except ImportError as exc:
    if "nms_impl" in str(exc):
        raise SystemExit(nms_build_message(include_arch=False)) from exc
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run official CLRNet DLA34 CULane checkpoint on one image."
    )
    parser.add_argument(
        "--config",
        default=str(OFFICIAL_CLRNET_ROOT / "configs/clrnet/clr_dla34_culane.py"),
        help="Official CLRNet CULane DLA34 config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights/culane_dla34.pth"),
        help="Trained CLRNet DLA34 checkpoint.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="CULane test image. If omitted, the script searches under workspace/data.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "clrnet_inference_test/outputs/single_image"),
        help="Directory for visualization and CULane text output.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--nms-thres", type=float, default=None)
    return parser.parse_args()


def find_first_image(data_root: Path) -> Path:
    search_roots = [
        data_root / "CULane",
        data_root,
        PROJECT_ROOT / "clrnet" / "data" / "CULane",
    ]
    exts = ("*.jpg", "*.jpeg", "*.png")
    for root in search_roots:
        if not root.exists():
            continue
        for ext in exts:
            for path in sorted(root.rglob(ext)):
                if path.is_file():
                    return path
    raise FileNotFoundError(
        "No image found. Pass --image /path/to/culane/test.jpg or extract/link CULane under workspace/data."
    )


def preprocess_bgr(image: np.ndarray, cfg) -> torch.Tensor:
    image = image[cfg.cut_height :, :, :]
    image = cv2.resize(image, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_CUBIC)
    # Official CULane val process ends GenerateLaneLine with float32 / 255.0.
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image.contiguous()


def lane_to_culane_line(lane, cfg) -> str:
    ys = np.arange(270, 590, 8) / cfg.ori_img_h
    xs = lane(ys)
    valid_mask = (xs >= 0) & (xs < 1)
    lane_xs = xs[valid_mask] * cfg.ori_img_w
    lane_ys = ys[valid_mask] * cfg.ori_img_h
    lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
    return " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(lane_xs, lane_ys))


def draw_lanes(image: np.ndarray, lanes, cfg) -> np.ndarray:
    vis = image.copy()
    for lane in lanes:
        points = lane.to_array(cfg)
        if len(points) < 2:
            continue
        pts = points.astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 255, 0), 2)
    return vis


def main():
    args = parse_args()
    if args.device is None:
        args.device = default_device_arg()
    cfg = Config.fromfile(args.config)
    cfg.backbone.pretrained = False
    if args.conf_threshold is not None:
        cfg.test_parameters.conf_threshold = args.conf_threshold
    if args.nms_thres is not None:
        cfg.test_parameters.nms_thres = args.nms_thres

    image_path = Path(args.image) if args.image else find_first_image(PROJECT_ROOT / "data")
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    device = resolve_device(args.device)
    model = build_net(cfg).to(device)
    load_checkpoint_for_inference(model, checkpoint)
    model.eval()

    tensor = preprocess_bgr(image, cfg).to(device)
    with torch.no_grad():
        output = model(tensor)
        lanes = model.heads.get_lanes(output)[0]

    stem = image_path.stem
    vis_path = output_dir / f"{stem}_clrnet_dla34.jpg"
    txt_path = output_dir / f"{stem}.lines.txt"

    vis = draw_lanes(image, lanes, cfg)
    cv2.imwrite(str(vis_path), vis)

    culane_lines = [lane_to_culane_line(lane, cfg) for lane in lanes]
    culane_lines = [line for line in culane_lines if line]
    txt_path.write_text("\n".join(culane_lines), encoding="utf-8")

    print(f"image={image_path}")
    print(f"checkpoint={checkpoint}")
    print(f"device={device}")
    print(f"lanes={len(lanes)}")
    print(f"visualization={vis_path}")
    print(f"culane_output={txt_path}")


if __name__ == "__main__":
    main()
