"""
Compact instance masks for evaluation (no pycocotools dependency).

A mask is stored as CropMask(x0, y0, crop) : a bool array covering only its bounding box in original-image
pixels. Mask IoU between two CropMasks only touches the overlap of their boxes, so evaluating 1,024 test
images x 100 detections stays in memory (full 608 x 608 masks would need tens of GB).

common.metrics.evaluate_detections is reused unchanged: per image, `boxes` is a numpy object array of
CropMask and the IoU function is mask_iou. Matching, `ignore` handling and AP are identical to detection.
"""

import cv2
import numpy as np


class CropMask:
    __slots__ = ("x0", "y0", "crop", "area")

    def __init__(self, x0, y0, crop):
        self.x0, self.y0 = int(x0), int(y0)
        self.crop = np.ascontiguousarray(crop, dtype=bool)
        self.area = int(self.crop.sum())

    @property
    def x1(self):
        return self.x0 + self.crop.shape[1]

    @property
    def y1(self):
        return self.y0 + self.crop.shape[0]

    @classmethod
    def from_full(cls, full):
        """bool [H, W] -> CropMask over the tight bounding box (an empty mask keeps a 0 x 0 crop)."""
        ys, xs = np.nonzero(full)
        if len(ys) == 0:
            return cls(0, 0, np.zeros((0, 0), bool))
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return cls(x0, y0, full[y0:y1, x0:x1])

    def to_full(self, h, w):
        out = np.zeros((h, w), bool)
        if self.area:
            out[self.y0:self.y1, self.x0:self.x1] = self.crop[: h - self.y0, : w - self.x0]
        return out


def as_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for k, it in enumerate(items):
        arr[k] = it
    return arr


def mask_iou_pair(a, b):
    if a.area == 0 or b.area == 0:
        return 0.0
    x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(np.logical_and(a.crop[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0],
                               b.crop[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]).sum())
    return inter / float(a.area + b.area - inter)


def mask_iou(det, gts):
    """IoU function for common.metrics.evaluate_detections: one CropMask vs an object array of CropMask."""
    return np.array([mask_iou_pair(det, g) for g in gts], dtype=np.float64)


def paste_roi_mask(prob, box, h, w, thr):
    """RoI mask probability [M, M] + box xyxy (original px) -> CropMask (torchvision paste_masks_in_image idea)."""
    x0, y0 = int(np.floor(max(box[0], 0))), int(np.floor(max(box[1], 0)))
    x1, y1 = int(np.ceil(min(box[2] + 1, w))), int(np.ceil(min(box[3] + 1, h)))
    if x1 <= x0 or y1 <= y0:
        return CropMask(0, 0, np.zeros((0, 0), bool))
    crop = cv2.resize(prob.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) >= thr
    return CropMask(x0, y0, crop)


def full_mask_to_original(prob, img_size, w, h, letterbox_params, thr, box=None):
    """Mask probability over the whole letterboxed input (any resolution [h', w'] proportional to img_size)
    -> CropMask in original pixels. If `box` (original px) is given, pixels outside it are cleared
    (YOLACT / YOLO-seg crop the assembled mask with the predicted box)."""
    scale, px, py, rw, rh = letterbox_params(w, h, img_size)
    r = prob.shape[0] / float(img_size)
    sub = prob[int(round(py * r)):int(round((py + rh) * r)), int(round(px * r)):int(round((px + rw) * r))]
    full = cv2.resize(sub.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) >= thr
    if box is not None:
        x0, y0 = int(max(np.floor(box[0]), 0)), int(max(np.floor(box[1]), 0))
        x1, y1 = int(min(np.ceil(box[2] + 1), w)), int(min(np.ceil(box[3] + 1), h))
        clip = np.zeros_like(full)
        clip[y0:y1, x0:x1] = True
        full &= clip
    return CropMask.from_full(full)


def semantic_confusion(pred_cls, gt_cls, num_classes, conf=None, void=255):
    """Accumulate a (C + 1) x (C + 1) confusion matrix (index 0 = background); void pixels are skipped."""
    n = num_classes + 1
    if conf is None:
        conf = np.zeros((n, n), np.int64)
    valid = gt_cls != void
    conf += np.bincount(n * gt_cls[valid].astype(np.int64) + pred_cls[valid], minlength=n * n).reshape(n, n)
    return conf


def miou_from_confusion(conf):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - np.diag(conf)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou, float(np.nanmean(iou))
