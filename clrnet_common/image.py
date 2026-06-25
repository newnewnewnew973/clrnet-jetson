"""Image preprocessing and lane formatting shared by CLRNet scripts."""

from pathlib import Path

import cv2
import numpy as np
import torch

from .runtime import PROJECT_ROOT


def find_first_image(data_root: Path) -> Path:
    """Return the first image found under the usual CULane locations."""
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
        "No image found. Pass --image /path/to/culane/test.jpg "
        "or extract/link CULane under data/."
    )


def preprocess_bgr(image: np.ndarray, cfg) -> torch.Tensor:
    """Apply the CLRNet CULane validation image preprocessing."""
    image = image[cfg.cut_height :, :, :]
    image = cv2.resize(image, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image.contiguous()


def lane_to_culane_line(lane, cfg) -> str:
    """Convert one CLRNet Lane object to one CULane prediction line."""
    ys = np.arange(270, 590, 8) / cfg.ori_img_h
    xs = lane(ys)
    valid_mask = (xs >= 0) & (xs < 1)
    lane_xs = xs[valid_mask] * cfg.ori_img_w
    lane_ys = ys[valid_mask] * cfg.ori_img_h
    lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
    return " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(lane_xs, lane_ys))


def draw_lanes(image: np.ndarray, lanes, cfg) -> np.ndarray:
    """Draw decoded CLRNet lanes on a BGR image."""
    vis = image.copy()
    for lane in lanes:
        points = lane.to_array(cfg)
        if len(points) < 2:
            continue
        pts = points.astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 255, 0), 2)
    return vis
