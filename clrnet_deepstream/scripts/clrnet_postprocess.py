"""Lightweight CLRNet postprocess for DeepStream overlay demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CulaneLayout:
    ori_img_w: int = 1640
    ori_img_h: int = 590
    img_w: int = 800
    img_h: int = 320
    cut_height: int = 270
    max_lanes: int = 4
    conf_threshold: float = 0.4
    duplicate_x_threshold: float = 0.05


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp[:, 1] / exp.sum(axis=1)


def is_duplicate(candidate: np.ndarray, selected: list[np.ndarray], threshold: float) -> bool:
    candidate_xs = candidate[6:]
    candidate_valid = (candidate_xs >= 0.0) & (candidate_xs <= 1.0)
    for lane in selected:
        lane_xs = lane[6:]
        valid = candidate_valid & (lane_xs >= 0.0) & (lane_xs <= 1.0)
        if valid.sum() < 8:
            continue
        if np.mean(np.abs(candidate_xs[valid] - lane_xs[valid])) < threshold:
            return True
    return False


def select_lane_predictions(predictions: np.ndarray, layout: CulaneLayout) -> list[np.ndarray]:
    rows = predictions[0] if predictions.ndim == 3 else predictions
    scores = softmax_positive(rows[:, :2])
    candidate_indices = np.flatnonzero(scores >= layout.conf_threshold)
    candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    selected: list[np.ndarray] = []
    for index in candidate_indices:
        lane = rows[index].copy()
        if is_duplicate(lane, selected, layout.duplicate_x_threshold):
            continue
        selected.append(lane)
        if len(selected) >= layout.max_lanes:
            break
    return selected


def prediction_to_points(lane: np.ndarray, layout: CulaneLayout) -> np.ndarray:
    lane_xs = lane[6:].copy()
    prior_ys = np.linspace(1.0, 0.0, num=lane_xs.shape[0], dtype=np.float32)
    n_strips = lane_xs.shape[0] - 1

    start = min(max(0, int(round(float(lane[2]) * n_strips))), n_strips)
    length = int(round(float(lane[5]) * n_strips))
    end = min(start + length - 1, len(prior_ys) - 1)

    if end + 1 < len(lane_xs):
        lane_xs[end + 1 :] = -2.0

    prefix = lane_xs[:start]
    prefix_valid = (prefix >= 0.0) & (prefix <= 1.0)
    if len(prefix_valid):
        keep_until_invalid = prefix_valid[::-1].cumprod()[::-1].astype(bool)
        lane_xs[:start][~keep_until_invalid] = -2.0

    valid = lane_xs >= 0.0
    xs = lane_xs[valid][::-1]
    ys = prior_ys[valid][::-1]
    if len(xs) < 2:
        return np.empty((0, 2), dtype=np.int32)

    ys = (ys * (layout.ori_img_h - layout.cut_height) + layout.cut_height) / layout.ori_img_h
    points = np.stack([xs * layout.ori_img_w, ys * layout.ori_img_h], axis=1)
    return points.astype(np.int32)


def decode_lane_points(predictions: np.ndarray, layout: CulaneLayout) -> list[np.ndarray]:
    lanes = select_lane_predictions(predictions, layout)
    points = [prediction_to_points(lane, layout) for lane in lanes]
    return [lane_points for lane_points in points if len(lane_points) >= 2]
