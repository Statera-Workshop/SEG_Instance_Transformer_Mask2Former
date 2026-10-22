"""
YOLO11-seg builder (seg task) - YOLO11 (detectors/yolo11, unchanged) + the Ultralytics Segment head.

Reference: Ultralytics release (2024, ultralytics/cfg/models/11/yolo11-seg.yaml: head `Segment, [nc, 32, 256]`;
ultralytics/nn/modules/head.py Segment, block.py Proto; utils/loss.py v8SegmentationLoss; utils/ops.py
process_mask / crop_mask), AGPL-3.0. Reimplemented here on top of the framework's YOLO11 (same blocks).

    Proto      on P3 (neck stride 8): Conv 3x3 (c_) -> ConvTranspose 2x2 s2 (c_) -> Conv 3x3 (c_) -> Conv 1x1 (32)
               -> 32 prototypes at stride 4 (152 x 152 for a 608 input); c_ = npr 256 x width (n: 64)
    cv4        per level: Conv 3x3 (c4) -> Conv 3x3 (c4) -> Conv2d 1x1 (32) mask coefficients, c4 = max(P3 ch / 4, 32)
    mask       sigmoid(coefficients . prototypes), cropped with the box
    loss       v8DetectionLoss (detectors/yolov8/loss.Criterion, unchanged) + v8SegmentationLoss mask term:
               per positive anchor, BCE(coef . proto, gt mask of the assigned object at proto resolution),
               averaged inside the assigned gt box (crop_mask), divided by the normalised box area, summed and
               divided by the number of positive anchors; gain = box gain 7.5 (Ultralytics loss[1] *= hyp.box)

Raw contract: YOLO11 rows [B, M, 5 + C] (one per anchor); aux = coefficients [B, M, 32] + prototypes.
decode_masks returns kind "full" at proto resolution; tasks/seg_task.py upsamples, crops with the box and
thresholds at 0.5 (process_mask order: crop at proto scale then upsample -> here upsample then crop, same mask
up to boundary pixels).

[Unified] as YOLO11 (no pretrained weights, common init, flip + YOLO-paper augmentation only, Adam, common
post-processing). Ground-truth masks are the dataset's instance masks downsampled to proto resolution with
nearest interpolation (Ultralytics rasterises polygons at 1/4 resolution; overlap_mask=False path, one mask
per object). Ultralytics also multiplies the summed loss by the batch size; the framework's v8 criterion does
not, so all four terms keep the Ultralytics relative weights.
"""

import torch


NM, NPR = 32, 256


def split_seg_targets(targets, batch_size, img_size):
    """{"boxes": [N, 6], "masks": [N, S, S]} -> per-image dicts (boxes xyxy px, labels 0..C-1, masks)."""
    t_all, m_all = targets["boxes"], targets["masks"]
    out = []
    for b in range(batch_size):
        sel = t_all[:, 0] == b
        t, m = t_all[sel], m_all[sel]
        cxcywh = t[:, 2:6] * img_size
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
        out.append({"labels": t[keep, 1].long(), "boxes": boxes[keep], "masks": m[keep]})
    return out
