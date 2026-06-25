"""Small subset of mmcv.cnn required by official CLRNet inference/evaluation.

The implemented ConvModule covers CLRNet's Conv2d + optional BN + optional ReLU
usage. Other MMCV ConvModule features intentionally raise NotImplementedError.
"""

import torch.nn as nn


DEFAULT_ACT_CFG = {"type": "ReLU"}
_DEFAULT_ACT_CFG_SENTINEL = object()


class ConvModule(nn.Module):
    """Conv-Norm-Activation block for the CLRNet configs used in this project."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=_DEFAULT_ACT_CFG_SENTINEL,
        inplace=True,
        **kwargs,
    ):
        super().__init__()
        if act_cfg is _DEFAULT_ACT_CFG_SENTINEL:
            act_cfg = DEFAULT_ACT_CFG
        if conv_cfg is not None:
            raise NotImplementedError("Only default Conv2d is supported")

        with_norm = norm_cfg is not None
        if bias == "auto":
            bias = not with_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.with_norm = with_norm
        if with_norm:
            norm_type = norm_cfg.get("type", "BN")
            if norm_type not in ("BN", "BN2d"):
                raise NotImplementedError(f"Unsupported norm type: {norm_type}")
            self.bn = nn.BatchNorm2d(out_channels)

        self.with_activation = act_cfg is not None
        if self.with_activation:
            act_type = act_cfg.get("type", "ReLU")
            if act_type != "ReLU":
                raise NotImplementedError(f"Unsupported activation type: {act_type}")
            self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x


__all__ = ["ConvModule"]
