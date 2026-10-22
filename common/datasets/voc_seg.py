"""
Instance segmentation dataset in VOC layout (FL36-Seg, VOC2007 segmentation subset).

Files per image id:
    JPEGImages/<id>.jpg
    Annotations/<id>.xml              boxes and classes (same files as detection)
    SegmentationObject/<id>.png       k = k-th <object> of the XML (1-based), 0 = background, 255 = void
    ImageSets/Segmentation/<split>.txt

Output contract (extends VOCDetection):
    image   : FloatTensor [3, S, S], RGB, letterboxed, 0..1
    targets : dict  boxes [N, 6] = [batch_index, class, x, y, w, h] normalised  (same as detection)
                    masks [N, S, S] uint8 {0, 1}, letterboxed, row-aligned with boxes

Boxes come from the XML, not from the masks, so the box branch is trained on exactly the detection
targets. Void (255) counts as background in the instance masks.

[Unified] Training augmentation identical to VOCDetection (profile v1.1: HSV jitter, scale / translate
jitter 0.2, hflip 0.5) with the same random-number order, applied to masks with nearest interpolation.
An instance is dropped by the detection rule (box < 25 % of its area or < 2 px after clipping) or when its
mask becomes empty.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.datasets.voc import VOCDetection, hsv_jitter, letterbox, letterbox_params, to_tensor


def load_instance_masks(png_path, n_objects):
    """SegmentationObject PNG -> bool [n_objects, H, W] (instance k <-> k-th XML object)."""
    ids = np.array(Image.open(png_path))           # palette PNG -> index array
    if ids.ndim == 3:
        ids = ids[..., 0]
    return np.stack([ids == k + 1 for k in range(n_objects)], 0) if n_objects else \
        np.zeros((0,) + ids.shape, bool)


def letterbox_masks(masks, size):
    """bool [N, H, W] -> uint8 [N, S, S] with the same geometry as letterbox() (nearest, pad 0)."""
    n, h, w = masks.shape
    scale, px, py, rw, rh = letterbox_params(w, h, size)
    out = np.zeros((n, size, size), np.uint8)
    for k in range(n):
        out[k, py:py + rh, px:px + rw] = cv2.resize(masks[k].astype(np.uint8), (rw, rh),
                                                    interpolation=cv2.INTER_NEAREST)
    return out


def random_scale_translate_masks(arr, boxes, masks, jitter, fill=128):
    """Same transform and random draws as voc.random_scale_translate, applied to image, boxes and masks."""
    size = arr.shape[0]
    sc = np.random.uniform(1.0 - jitter, 1.0 + jitter)
    tx, ty = np.random.uniform(-jitter, jitter, 2) * size
    c = (1.0 - sc) * size / 2.0
    m = np.array([[sc, 0, c + tx], [0, sc, c + ty]], dtype=np.float32)
    out = cv2.warpAffine(arr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(fill, fill, fill))
    if len(boxes) == 0:
        return out, boxes, masks, np.zeros(0, bool)
    mk = np.stack([cv2.warpAffine(x, m, (size, size), flags=cv2.INTER_NEAREST, borderValue=0) for x in masks], 0)
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]] * sc + c + tx
    b[:, [1, 3]] = b[:, [1, 3]] * sc + c + ty
    area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = np.clip(b, 0, size)
    area1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = (b[:, 2] - b[:, 0] > 2) & (b[:, 3] - b[:, 1] > 2) & (area1 >= 0.25 * np.maximum(area0, 1e-6))
    keep &= mk.reshape(len(mk), -1).any(1)
    return out, b, mk, keep


class VOCInstanceSeg(VOCDetection):
    def __init__(self, root, split, img_size, class_names, train=False, hflip_prob=0.0, max_samples=None, aug=None):
        # same attributes as VOCDetection; the id list comes from ImageSets/Segmentation
        self.root = Path(root)
        self.img_size = int(img_size)
        self.aug = aug if train else None
        self.class_names = list(class_names)
        # [Unified] 2026-10-16 patch 15: keys lower-cased like the XML names below (FL36 letters "A"-"Z" were dropped)
        self.cls_to_id = {n.lower(): i for i, n in enumerate(self.class_names)}
        assert len(self.cls_to_id) == len(self.class_names), "class names must be unique ignoring case"
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        self.ids = [x.strip() for x in split_file.read_text().split() if x.strip()]
        if max_samples:
            self.ids = self.ids[:max_samples]
        self._ann_cache = {}

    def mask_path(self, i):
        return self.root / "SegmentationObject" / f"{self.ids[i]}.png"

    def annotation(self, i):
        """Detection annotation + `object_index` (position of each kept object in the XML) for the masks."""
        if i in self._ann_cache:
            return self._ann_cache[i]
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.root / "Annotations" / f"{self.ids[i]}.xml").getroot()
        boxes, labels, difficult, obj_idx = [], [], [], []
        for k, obj in enumerate(tree.iter("object")):
            name = obj.find("name").text.strip().lower()
            if name not in self.cls_to_id:
                continue
            d = obj.find("difficult")
            bb = obj.find("bndbox")
            boxes.append([float(bb.find(t).text) - 1.0 for t in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(self.cls_to_id[name])
            difficult.append(bool(int(d.text)) if d is not None else False)
            obj_idx.append(k)
        ann = {"boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
               "labels": np.array(labels, dtype=np.int64),
               "ignore": np.array(difficult, dtype=bool),
               "object_index": np.array(obj_idx, dtype=np.int64),
               "n_xml_objects": len(list(tree.iter("object")))}
        self._ann_cache[i] = ann
        return ann

    def masks(self, i):
        """bool [N, H, W] in original pixels, row-aligned with annotation(i)['boxes']."""
        ann = self.annotation(i)
        allm = load_instance_masks(self.mask_path(i), ann["n_xml_objects"])
        return allm[ann["object_index"]]

    def __getitem__(self, i):
        image = Image.open(self.image_path(i)).convert("RGB")
        ann = self.annotation(i)
        keep = ~ann["ignore"] if self.train else np.ones(len(ann["labels"]), bool)
        boxes, labels = ann["boxes"][keep], ann["labels"][keep]
        masks = letterbox_masks(self.masks(i)[keep], self.img_size)

        image, scale, px, py = letterbox(image, self.img_size)
        s = float(self.img_size)
        b = boxes * scale
        if len(b):
            b[:, [0, 2]] += px
            b[:, [1, 3]] += py
        if self.aug:
            arr = hsv_jitter(np.asarray(image), self.aug["hue"], self.aug["saturation"], self.aug["exposure"])
            arr, b, masks, keep = random_scale_translate_masks(arr, b, masks, self.aug["jitter"])
            b, labels, masks = b[keep], labels[keep], masks[keep]
            image = Image.fromarray(arr)
        targets = torch.zeros((len(b), 6), dtype=torch.float32)
        if len(b):
            targets[:, 1] = torch.from_numpy(labels).float()
            targets[:, 2] = torch.from_numpy((b[:, 0] + b[:, 2]) / 2 / s)
            targets[:, 3] = torch.from_numpy((b[:, 1] + b[:, 3]) / 2 / s)
            targets[:, 4] = torch.from_numpy((b[:, 2] - b[:, 0]) / s)
            targets[:, 5] = torch.from_numpy((b[:, 3] - b[:, 1]) / s)
            targets[:, 2:6] = targets[:, 2:6].clamp(0.0, 1.0)
        m = torch.from_numpy(np.ascontiguousarray(masks)).reshape(-1, self.img_size, self.img_size)

        img = to_tensor(image)
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img = torch.flip(img, [-1])
            targets[:, 2] = 1.0 - targets[:, 2]
            m = torch.flip(m, [-1])
        return i, img, {"boxes": targets, "masks": m}

    @staticmethod
    def collate_fn(batch):
        idx, images, targets = zip(*batch)
        boxes, masks = [], []
        for b, t in enumerate(targets):
            if t["boxes"].numel():
                bt = t["boxes"].clone()
                bt[:, 0] = b
                boxes.append(bt)
                masks.append(t["masks"])
        S = images[0].shape[-1]
        return list(idx), torch.stack(images, 0), {
            "boxes": torch.cat(boxes, 0) if boxes else torch.zeros((0, 6)),
            "masks": torch.cat(masks, 0) if masks else torch.zeros((0, S, S), dtype=torch.uint8)}
