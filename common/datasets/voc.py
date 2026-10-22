"""
PASCAL VOC2007 dataset for the image track.

Output contract (same as the legacy ImageYoloDataset):
    image   : FloatTensor [3, S, S], RGB, letterboxed, 0..1
    targets : FloatTensor [N, 6] = [batch_index, class, x, y, w, h] normalized

Training excludes "difficult" objects (VOC convention). Evaluation keeps them
as `ignore` ground truths.

[Unified] Training augmentation (profile v1.1, `aug`): the basic augmentation of the original
YOLO papers, shared by every version -
    YOLOv1 paper : random scaling and translation of up to 20 % of the image size,
                   random exposure and saturation of up to a factor 1.5 in HSV
    YOLOv2 / v3  : Darknet cfg hue = .1, saturation = 1.5, exposure = 1.5
plus the common horizontal flip. Version-specific strong augmentation (Mosaic, MixUp, ...) is
not used. Boxes follow the same affine transform; boxes left with less than 25 % of their area
(or narrower than 2 px) after clipping are dropped.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def letterbox_params(w, h, size):
    scale = min(size / w, size / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    return scale, (size - rw) // 2, (size - rh) // 2, rw, rh


def letterbox(image, size):
    """PIL RGB -> (letterboxed PIL, scale, pad_x, pad_y). Gray (128) padding, legacy."""
    w, h = image.size
    scale, px, py, rw, rh = letterbox_params(w, h, size)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(image.resize((rw, rh), Image.BILINEAR), (px, py))
    return canvas, scale, px, py


def _rand_scale(s):
    """Darknet rand_scale: a factor in [1, s], inverted with probability 0.5."""
    f = np.random.uniform(1.0, s)
    return f if np.random.random() < 0.5 else 1.0 / f


def hsv_jitter(arr, hue, saturation, exposure):
    """uint8 RGB -> uint8 RGB with random hue shift and saturation / exposure scaling."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-hue, hue) * 180.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * _rand_scale(saturation), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * _rand_scale(exposure), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_scale_translate(arr, boxes, jitter, fill=128):
    """Scale about the centre by [1 - jitter, 1 + jitter] and shift by up to jitter * size."""
    size = arr.shape[0]
    sc = np.random.uniform(1.0 - jitter, 1.0 + jitter)
    tx, ty = np.random.uniform(-jitter, jitter, 2) * size
    c = (1.0 - sc) * size / 2.0
    m = np.array([[sc, 0, c + tx], [0, sc, c + ty]], dtype=np.float32)
    out = cv2.warpAffine(arr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(fill, fill, fill))
    if len(boxes) == 0:
        return out, boxes, np.zeros(0, bool)
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]] * sc + c + tx
    b[:, [1, 3]] = b[:, [1, 3]] * sc + c + ty
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = np.clip(b, 0, size)
    area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2) & (area1 >= 0.25 * np.maximum(area0, 1e-6))
    return out, b, keep


def to_tensor(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class VOCDetection(Dataset):
    def __init__(self, root, split, img_size, class_names, train=False, hflip_prob=0.0, max_samples=None, aug=None):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.aug = aug if train else None     # [Unified] training augmentation (see module docstring)
        self.class_names = list(class_names)
        # [Unified] 2026-10-16 patch 15: keys lower-cased like the XML names below (FL36 letters "A"-"Z" were dropped)
        self.cls_to_id = {n.lower(): i for i, n in enumerate(self.class_names)}
        assert len(self.cls_to_id) == len(self.class_names), "class names must be unique ignoring case"
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        split_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        self.ids = [x.strip() for x in split_file.read_text().split() if x.strip()]
        if max_samples:
            self.ids = self.ids[:max_samples]
        self._ann_cache = {}

    def __len__(self):
        return len(self.ids)

    def image_path(self, i):
        return self.root / "JPEGImages" / f"{self.ids[i]}.jpg"

    def annotation(self, i):
        """Original-pixel xyxy boxes, labels, difficult flags."""
        if i in self._ann_cache:
            return self._ann_cache[i]
        tree = ET.parse(self.root / "Annotations" / f"{self.ids[i]}.xml").getroot()
        boxes, labels, difficult = [], [], []
        for obj in tree.iter("object"):
            name = obj.find("name").text.strip().lower()
            if name not in self.cls_to_id:
                continue
            d = obj.find("difficult")
            bb = obj.find("bndbox")
            box = [float(bb.find(k).text) - 1.0 for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append(box)
            labels.append(self.cls_to_id[name])
            difficult.append(bool(int(d.text)) if d is not None else False)
        ann = {
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.array(labels, dtype=np.int64),
            "ignore": np.array(difficult, dtype=bool),
        }
        self._ann_cache[i] = ann
        return ann

    def __getitem__(self, i):
        image = Image.open(self.image_path(i)).convert("RGB")
        w, h = image.size
        ann = self.annotation(i)
        keep = ~ann["ignore"] if self.train else np.ones(len(ann["labels"]), bool)
        boxes, labels = ann["boxes"][keep], ann["labels"][keep]

        image, scale, px, py = letterbox(image, self.img_size)
        s = float(self.img_size)
        b = boxes * scale
        if len(b):
            b[:, [0, 2]] += px
            b[:, [1, 3]] += py
        if self.aug:
            arr = hsv_jitter(np.asarray(image), self.aug["hue"], self.aug["saturation"], self.aug["exposure"])
            arr, b, keep = random_scale_translate(arr, b, self.aug["jitter"])
            b, labels = b[keep], labels[keep]
            image = Image.fromarray(arr)
        targets = torch.zeros((len(b), 6), dtype=torch.float32)
        if len(b):
            targets[:, 1] = torch.from_numpy(labels).float()
            targets[:, 2] = torch.from_numpy((b[:, 0] + b[:, 2]) / 2 / s)
            targets[:, 3] = torch.from_numpy((b[:, 1] + b[:, 3]) / 2 / s)
            targets[:, 4] = torch.from_numpy((b[:, 2] - b[:, 0]) / s)
            targets[:, 5] = torch.from_numpy((b[:, 3] - b[:, 1]) / s)
            targets[:, 2:6] = targets[:, 2:6].clamp(0.0, 1.0)

        img = to_tensor(image)
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img = torch.flip(img, [-1])
            targets[:, 2] = 1.0 - targets[:, 2]
        return i, img, targets

    @staticmethod
    def collate_fn(batch):
        idx, images, targets = zip(*batch)
        out = []
        for b, t in enumerate(targets):
            if t.numel():
                t = t.clone()
                t[:, 0] = b
                out.append(t)
        targets = torch.cat(out, 0) if out else torch.zeros((0, 6))
        return list(idx), torch.stack(images, 0), targets
