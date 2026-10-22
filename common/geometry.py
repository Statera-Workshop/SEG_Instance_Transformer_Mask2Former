"""
Common geometry: IoU and NMS used by every detector.

2D (image)      boxes xyxy
BEV (rotated)   boxes (x, y, w, l, yaw)   -> shapely polygons
3D              boxes (x, y, z, h, w, l, yaw), z = bottom of the box

One implementation, used by training-free parts of all tasks:
post-processing (NMS) and evaluation (TP/FP matching).
"""

import numpy as np
import torch

try:
    from torchvision.ops import nms as _tv_nms
except Exception:  # torchvision is optional
    _tv_nms = None


# ------------------------------------------------------------------
# 2D
# ------------------------------------------------------------------
def box_iou_2d(box, boxes):
    """IoU of one xyxy box [4] against N boxes [N, 4] (numpy)."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-12)


def nms_2d(boxes, scores, iou_thr):
    """Greedy NMS on xyxy boxes (torch). Returns kept indices, score-sorted."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if _tv_nms is not None:
        return _tv_nms(boxes.float(), scores.float(), iou_thr)
    order = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        w = (torch.minimum(x2[i], x2[rest]) - torch.maximum(x1[i], x1[rest])).clamp(min=0)
        h = (torch.minimum(y2[i], y2[rest]) - torch.maximum(y1[i], y1[rest])).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        order = rest[iou <= iou_thr]
    return torch.stack(keep)


def batched_nms_2d(boxes, scores, labels, iou_thr):
    """Class-aware NMS via the coordinate-offset trick."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    offset = labels.to(boxes.dtype)[:, None] * (boxes.max() + 1)
    return nms_2d(boxes + offset, scores, iou_thr)


# ------------------------------------------------------------------
# Rotated BEV
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Evaluation IoU functions on metric LiDAR boxes [x, y, z, h, w, l, yaw]
# ------------------------------------------------------------------
