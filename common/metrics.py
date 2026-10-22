"""
Common detection metrics: Precision, Recall, AP, mAP.

One matching procedure for every task (PASCAL VOC protocol):
    1. per class, sort all detections of the split by score
    2. a detection is TP if its best-IoU ground truth (same class, same image)
       has IoU >= iou_thr and is not matched yet; otherwise FP
    3. ground truths flagged `ignore` (VOC "difficult") neither count as
       positives nor produce FP when matched
    4. AP = area under the interpolated precision/recall curve

The IoU function is the only thing that changes between tasks:
    image   -> 2D IoU
    complex -> rotated BEV IoU
    yolo3d  -> rotated BEV IoU and 3D IoU
"""

import numpy as np


def compute_ap(recall, precision, method="all"):
    """
    method="all" : all-point interpolation (VOC2010+, default)
    method="11pt": 11-point interpolation (official VOC2007)
    """
    if method == "11pt":
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p / 11.0
        return float(ap)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate_detections(dets, gts, class_names, iou_fn, iou_thr=0.5):
    """
    dets: list (per image) of dict(boxes [N, K], scores [N], labels [N])
    gts : list (per image) of dict(boxes [M, K], labels [M], ignore [M] bool)
    """
    results = {}
    for c, name in enumerate(class_names):
        gt_c = []
        npos = 0
        for g in gts:
            m = g["labels"] == c
            ign = g["ignore"][m] if "ignore" in g else np.zeros(m.sum(), bool)
            gt_c.append({"boxes": g["boxes"][m], "ignore": ign, "used": np.zeros(m.sum(), bool)})
            npos += int((~ign).sum())

        records = []
        for img_i, d in enumerate(dets):
            m = d["labels"] == c
            for box, score in zip(d["boxes"][m], d["scores"][m]):
                records.append((score, img_i, box))
        records.sort(key=lambda r: -r[0])

        tp = np.zeros(len(records))
        fp = np.zeros(len(records))
        for k, (_, img_i, box) in enumerate(records):
            g = gt_c[img_i]
            if len(g["boxes"]) == 0:
                fp[k] = 1
                continue
            ious = iou_fn(box, g["boxes"])
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                if g["ignore"][j]:
                    continue
                if not g["used"][j]:
                    tp[k] = 1
                    g["used"][j] = True
                else:
                    fp[k] = 1
            else:
                fp[k] = 1

        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        rec = tpc / max(npos, 1)
        prec = tpc / np.maximum(tpc + fpc, np.finfo(np.float64).eps)
        results[name] = {
            "AP": compute_ap(rec, prec, "all") if npos > 0 else 0.0,
            "AP_11pt": compute_ap(rec, prec, "11pt") if npos > 0 else 0.0,
            "precision": float(prec[-1]) if len(prec) else 0.0,
            "recall": float(rec[-1]) if len(rec) else 0.0,
            "num_gt": npos,
            "num_det": len(records),
        }

    valid = [r for r in results.values() if r["num_gt"] > 0]
    summary = {
        "mAP": float(np.mean([r["AP"] for r in valid])) if valid else 0.0,
        "mAP_11pt": float(np.mean([r["AP_11pt"] for r in valid])) if valid else 0.0,
        "precision": float(np.mean([r["precision"] for r in valid])) if valid else 0.0,
        "recall": float(np.mean([r["recall"] for r in valid])) if valid else 0.0,
    }
    return {"per_class": results, "summary": summary, "iou_thr": iou_thr}


def print_results(title, res):
    print(f"\n{title}  (IoU >= {res['iou_thr']})")
    print(f"{'class':<14}{'AP':>8}{'AP11':>8}{'P':>8}{'R':>8}{'#GT':>8}{'#Det':>8}")
    for name, r in res["per_class"].items():
        if r["num_gt"] == 0 and r["num_det"] == 0:
            continue
        print(f"{name:<14}{r['AP']:>8.4f}{r['AP_11pt']:>8.4f}{r['precision']:>8.4f}"
              f"{r['recall']:>8.4f}{r['num_gt']:>8d}{r['num_det']:>8d}")
    s = res["summary"]
    print(f"{'mean':<14}{s['mAP']:>8.4f}{s['mAP_11pt']:>8.4f}{s['precision']:>8.4f}{s['recall']:>8.4f}")
