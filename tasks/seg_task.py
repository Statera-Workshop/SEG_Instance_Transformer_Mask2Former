"""
Instance segmentation task (FL36-Seg / VOC2007 segmentation subset).

Model contract
--------------
Training  : loss, _ = model(images, {"boxes": [N, 6], "masks": [N, S, S] uint8})
Inference : model(images) -> (raw, aux)
    raw  [B, N, 5 + C]  same box contract as detection (cx cy w h, objectness, class probs; input px)
    aux  anything the model needs to build masks later
Masks     : model.decode_masks(aux, b, rows, boxes, labels) -> {"kind": "roi" | "full", "prob": [K, h, w]}
    b       image index in the batch
    rows    LongTensor [K]  rows of raw[b] kept by the common post-processing (score filter, top-k, NMS)
    boxes   [K, 4] xyxy input px, labels [K]
    kind "roi"  : prob is an M x M RoI mask aligned with each box (Mask R-CNN)
    kind "full" : prob covers the whole letterboxed input at any resolution proportional to the input
                  (YOLACT / YOLO-seg: assembled prototypes); it is cropped with the predicted box

Box post-processing is ImageTask's (conf 0.01, top-300, NMS 0.5, max 100), so box AP50 is directly
comparable with the detection stage. Masks are only decoded for the <= 100 survivors.

Metrics
-------
box AP50, mask AP50 (same VOC matching, IoU = mask IoU on CropMask, common.mask_ops), and
semantic mIoU derived from instances (detections with score >= 0.5 painted in ascending score order;
ground truth painted the same way; index 0 = background).
"""

import time

import numpy as np
import torch
from PIL import Image

from common import config as C
from common.datasets.voc import letterbox_params
from common.datasets.voc_seg import VOCInstanceSeg
from common.geometry import batched_nms_2d, box_iou_2d
from common.mask_ops import (CropMask, as_object_array, full_mask_to_original, mask_iou, miou_from_confusion,
                             paste_roi_mask, semantic_confusion)
from common.metrics import evaluate_detections, print_results
from common.progress import Progress
from common.trainer import make_loader

SEMANTIC_SCORE = 0.5


class SegTask:
    name = "seg"

    def __init__(self):
        self.spec = C.TASKS["seg"]
        self.class_names = list(self.spec["classes"])

    # ------------------------------------------------------------ data
    def train_set(self, args):
        return VOCInstanceSeg(C.SEG_ROOT, self.spec["train_split"], args.img_size, self.class_names, train=True,
                              hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train,
                              aug=C.PROFILE["image_aug"])

    def test_set(self, args):
        return VOCInstanceSeg(C.SEG_ROOT, self.spec["test_split"], args.img_size, self.class_names, train=False,
                              max_samples=args.max_test)

    # ------------------------------------------------------------ post-process
    @staticmethod
    def select(pred, conf_thr):
        """ImageTask.postprocess selection, also returning the kept row indices."""
        P = C.PROFILE
        cls_conf, cls_id = pred[:, 5:].max(1)
        score = pred[:, 4] * cls_conf
        rows = torch.nonzero(score >= conf_thr).squeeze(1)
        if rows.numel() > P["pre_nms_topk"]:
            rows = rows[score[rows].topk(P["pre_nms_topk"]).indices]
        p = pred[rows]
        boxes = torch.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                             p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
        k = batched_nms_2d(boxes, score[rows], cls_id[rows], P["nms_iou"])[: P["max_det"]]
        return rows[k], boxes[k], score[rows][k], cls_id[rows][k]

    def postprocess(self, model, out, orig_sizes, img_size, conf_thr):
        raw, aux = out
        thr = self.spec["mask_threshold"]
        res = []
        for b, (pred, (w, h)) in enumerate(zip(raw, orig_sizes)):
            rows, boxes_in, score, cls_id = self.select(pred, conf_thr)
            scale, px, py, _, _ = letterbox_params(w, h, img_size)
            boxes = boxes_in.clone()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / scale).clamp(0, w - 1)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / scale).clamp(0, h - 1)
            boxes_np = boxes.cpu().numpy()
            masks = []
            if rows.numel():
                dm = model.decode_masks(aux, b, rows, boxes_in, cls_id)
                prob = dm["prob"].float().cpu().numpy()
                for k in range(len(rows)):
                    if dm["kind"] == "roi":
                        masks.append(paste_roi_mask(prob[k], boxes_np[k], h, w, thr))
                    else:
                        masks.append(full_mask_to_original(prob[k], img_size, w, h, letterbox_params, thr,
                                                           box=boxes_np[k]))
            res.append({"boxes": boxes_np, "scores": score.cpu().numpy(), "labels": cls_id.cpu().numpy(),
                        "masks": as_object_array(masks)})
        return res

    # ------------------------------------------------------------ evaluation
    def gt(self, dataset, i):
        ann = dataset.annotation(i)
        full = dataset.masks(i)
        return {**{k: ann[k] for k in ("boxes", "labels", "ignore")},
                "masks": as_object_array([CropMask.from_full(m) for m in full]), "full": full}

    @staticmethod
    def paint(masks_full_or_crop, labels, scores, h, w, order_by_score):
        sem = np.zeros((h, w), np.int64)
        order = np.argsort(scores) if order_by_score else range(len(labels))
        for k in order:
            m = masks_full_or_crop[k]
            m = m.to_full(h, w) if isinstance(m, CropMask) else m
            sem[m] = int(labels[k]) + 1
        return sem

    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        dets, gts = [], []
        conf = None
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for idx, images, _ in loader:
            out = model(images.to(device, dtype=torch.float32))
            sizes = [Image.open(dataset.image_path(i)).size for i in idx]
            d = self.postprocess(model, out, sizes, args.img_size, C.PROFILE["eval_conf"])
            for i, di, (w, h) in zip(idx, d, sizes):
                g = self.gt(dataset, i)
                keep = di["scores"] >= SEMANTIC_SCORE
                pred_sem = self.paint(di["masks"][keep], di["labels"][keep], di["scores"][keep], h, w, True)
                gt_sem = self.paint(g.pop("full"), g["labels"], np.zeros(len(g["labels"])), h, w, False)
                conf = semantic_confusion(pred_sem, gt_sem, len(self.class_names), conf)
                gts.append(g)
                dets.append(di)
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        self.last = (dets, gts)                  # kept for diagnostics (tools/overfit_check_seg.py)
        split = getattr(args, "split_label", self.spec["test_split"])
        title = f"{self.spec['dataset'].upper()} {split} ({len(dataset)} images)"
        box_res = evaluate_detections(dets, gts, self.class_names, box_iou_2d, C.PROFILE["eval_iou"])
        print_results(title + " | box", box_res)
        mdets = [{"boxes": d["masks"], "scores": d["scores"], "labels": d["labels"]} for d in dets]
        mgts = [{"boxes": g["masks"], "labels": g["labels"], "ignore": g["ignore"]} for g in gts]
        cache = {}

        def cached_mask_iou(det, g):                 # same pairs at every threshold -> compute once
            key = (id(det), id(g))
            if key not in cache:
                cache[key] = mask_iou(det, g)
            return cache[key]

        mask_res = evaluate_detections(mdets, mgts, self.class_names, cached_mask_iou, C.PROFILE["eval_iou"])
        print_results(title + " | mask", mask_res)
        # [Unified] 2026-10-11: AP50 saturates on FL36 (every box TP is also a mask TP); report stricter
        # thresholds too. AP50:95 = mean over IoU 0.50, 0.55, ..., 0.95 (COCO thresholds, VOC 11-point-free AP).
        strict = {}
        for name, iou_fn, ds, gs in (("box", box_iou_2d, dets, gts), ("mask", cached_mask_iou, mdets, mgts)):
            aps = [evaluate_detections(ds, gs, self.class_names, iou_fn, t / 100)["summary"]["mAP"]
                   for t in range(50, 100, 5)]
            strict[name] = {"ap75": aps[5], "ap50_95": float(np.mean(aps))}
            print(f"{name:<5} AP75 {aps[5]:.4f}  AP50:95 {np.mean(aps):.4f}")
        iou, miou = miou_from_confusion(conf)
        print(f"Derived semantic mIoU (score >= {SEMANTIC_SCORE}, incl. background): {miou:.4f}")
        print(f"Inference speed incl. post-process and mask decoding: {fps:.1f} img/s")
        return {"metric": "mask AP50 / box AP50 (VOC matching, difficult ignored)", "images": len(dataset),
                "fps": fps, "mask_ap50": mask_res, "ap50": box_res, "strict": strict,
                "semantic": {"miou": miou, "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
                             "score_thr": SEMANTIC_SCORE}}
