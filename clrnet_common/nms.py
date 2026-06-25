"""Shared CLRNet NMS entry point."""

from .nms_loader import load_nms_impl


_nms_impl = load_nms_impl()


def nms(boxes, scores, overlap, top_k):
    return _nms_impl.nms_forward(boxes, scores, overlap, top_k)
