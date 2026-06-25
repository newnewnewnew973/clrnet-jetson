"""Small subset of mmcv.parallel required by CLRNet inference/evaluation.

The local scripts do not use MMCV's full data-parallel or DataContainer stack.
These definitions exist so official CLRNet dataset/model imports can run while
the scripts directly feed tensors to the model.
"""

import torch


class DataContainer:
    """Store metadata in the shape expected by CLRNet dataset code."""
    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs


class MMDataParallel(torch.nn.DataParallel):
    """Compatibility alias for imports; local scripts do not rely on it."""
    pass


def collate(batch, samples_per_gpu=1):
    """Minimal collate fallback; not a full MMCV DataContainer collator."""
    return torch.utils.data.default_collate(batch)


__all__ = ["DataContainer", "MMDataParallel", "collate"]
