"""Small subset of mmcv.runner required by CLRNet inference/evaluation."""


def auto_fp16(*decorator_args, **decorator_kwargs):
    """No-op auto_fp16 decorator for inference-only compatibility."""
    def decorator(func):
        return func

    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
        return decorator_args[0]
    return decorator


__all__ = ["auto_fp16"]
