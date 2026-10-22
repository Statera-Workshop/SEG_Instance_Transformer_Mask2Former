"""
Common experiment profile and paths.

Every train / evaluate script of every detector reads its settings
from here. Only the algorithm (detectors/<version>/) differs between runs.

Paths can be moved with one environment variable:
    export PHYSICAL_AI_ROOT=~/Physical_AI
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_AI_ROOT = Path(os.environ.get("PHYSICAL_AI_ROOT", "~/Physical_AI")).expanduser()
DATASETS_ROOT = PHYSICAL_AI_ROOT / "datasets"

RUNS_ROOT = FRAMEWORK_ROOT / "runs"

# ------------------------------------------------------------------
# OD-Small Benchmark Profile v1.0  (Mode A: controlled comparison)
# ------------------------------------------------------------------
PROFILE = {
    "name": "OD-Small v1.1",       # v1.1: Image training augmentation of the YOLO papers (image_aug)
    "seed": 777,
    "epochs": 30,   # [Release] FL36-Seg validation: 30 epochs
    "optimizer": "Adam",           # decided 2026-09-28 (legacy Complex/YOLO3D optimizer)
    "lr": 1e-4,   # [Release] FL36-Seg validation: Adam 1e-4
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_schedule": "constant",     # legacy StepLR(10) never fires within 10 epochs
    "effective_batch": 8,          # micro batch x gradient accumulation
    "hflip_prob": 0.5,             # common augmentation for every task
    # [Unified] profile v1.1 (2026-09-30): basic augmentation of the YOLOv1 / v2 papers for the Image
    # track (10 epochs from scratch with flip only memorised the training set: train cls acc 99.6 %,
    # test mAP50 < 0.04 in the 1/4-VOC convergence study). KITTI keeps its point-cloud augmentation.
    "image_aug": {"jitter": 0.2, "hue": 0.1, "saturation": 1.5, "exposure": 1.5},
    "eval_iou": 0.5,               # AP50 / mAP50
    "eval_conf": 0.01,             # low threshold -> full PR curve
    "nms_iou": 0.5,                # same NMS threshold for every task
    "pre_nms_topk": 300,
    "max_det": 100,
}

# Mode B (reference recipe): legacy optimizer of each project.
REFERENCE_RECIPE = {
    "image":   {"optimizer": "SGD",  "lr": 2.5e-4, "lr_schedule": "constant"},
    "complex": {"optimizer": "Adam", "lr": 1e-3,   "lr_schedule": "step10x0.8"},
    "yolo3d":  {"optimizer": "Adam", "lr": 1e-3,   "lr_schedule": "step10x0.8"},
}

# ------------------------------------------------------------------
# Task definitions (dataset side only - no algorithm here)
# ------------------------------------------------------------------
# FashionLetters36 classes: Fashion-MNIST 10 + EMNIST letters 26 (FL36-Seg uses the same 36 classes)
FL36_CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
                *[chr(ord("A") + i) for i in range(26)])


# [Unified] Segmentation stage (2026-10-11). Instance segmentation data in VOC layout:
#   JPEGImages/, Annotations/ (boxes, unchanged), SegmentationObject/ (instance id PNG: k = k-th XML object,
#   0 = background, 255 = void), SegmentationClass/ (class id + 1 PNG), ImageSets/Segmentation/<split>.txt.
# FL36-Seg (default) is FashionLetters36 regenerated with the same seed plus masks; VOC2007 is the check set
# (422 trainval / 210 test images with masks). Select with UNIFIED_SEG_DATASET=fl36seg | voc2007.
SEG_DATASET = os.environ.get("UNIFIED_SEG_DATASET", "fl36seg")
FL36SEG_ROOT = DATASETS_ROOT / "FashionLetters36_Seg"
if SEG_DATASET == "fl36seg":
    SEG_ROOT, SEG_CLASSES = FL36SEG_ROOT, FL36_CLASSES
elif SEG_DATASET == "voc2007":
    SEG_ROOT = DATASETS_ROOT / "VOC" / "VOCdevkit" / "VOC2007"
    SEG_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                   "tvmonitor")
else:
    raise ValueError(f"UNIFIED_SEG_DATASET must be fl36seg or voc2007, not {SEG_DATASET!r}")

TASKS = {
    "seg": {
        "dataset": SEG_DATASET,
        "classes": SEG_CLASSES,
        "img_size": 608,
        "micro_batch": 2,
        "train_split": "trainval",
        "test_split": "test",
        "mask_threshold": 0.5,       # mask probability -> binary
    },
}


def run_dir(task, version):
    return RUNS_ROOT / f"{task}_{version}"


def checkpoint_path(task, version, epochs):
    return run_dir(task, version) / "checkpoints" / f"{task}_{version}_{TASKS[task]['dataset']}_e{epochs:03d}.pth"
