"""Minimal MMCV compatibility layer for CLRNet inference/evaluation.

This is not a full MMCV replacement. It only exposes the symbols reached by the
official CLRNet DLA34 CULane inference and evaluation path on this Jetson setup.
"""

from .io import dump, load


def jit(*jit_args, **jit_kwargs):
    """No-op stand-in for mmcv.jit used by imported CLRNet utilities."""
    def decorator(func):
        return func

    if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
        return jit_args[0]
    return decorator


__all__ = ["dump", "jit", "load"]
