"""INT8 calibration utilities for TensorRT CLRNet engines."""

from pathlib import Path

import tensorrt as trt
import torch


class CULaneEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT entropy calibrator backed by the official CULane pipeline."""

    def __init__(
        self,
        dataset,
        cache_path: Path,
        sample_count: int,
    ) -> None:
        super().__init__()
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        self.dataset = dataset
        self.cache_path = cache_path
        self.indices = self._make_indices(len(dataset), sample_count)
        self.current_batch: torch.Tensor | None = None
        self.cursor = 0

    @staticmethod
    def _make_indices(dataset_size: int, sample_count: int) -> list[int]:
        if dataset_size < 1:
            raise ValueError("dataset is empty; cannot calibrate INT8 engine")
        count = min(sample_count, dataset_size)
        if count == 1:
            return [0]
        step = (dataset_size - 1) / float(count - 1)
        return [round(index * step) for index in range(count)]

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.cursor >= len(self.indices):
            return None

        sample = self.dataset[self.indices[self.cursor]]
        self.cursor += 1
        self.current_batch = sample["img"].unsqueeze(0).contiguous().cuda()
        return [int(self.current_batch.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(bytes(cache))
