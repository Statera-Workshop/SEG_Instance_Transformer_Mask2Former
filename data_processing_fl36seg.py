#!/usr/bin/env python3
"""
FL36-Seg: FashionLetters36 regenerated with the same seed, plus instance / class masks.

    python data_processing_fl36seg.py                           # downloads Fashion-MNIST + EMNIST
    python data_processing_fl36seg.py --emnist-zip ~/Downloads/gzip.zip

The original generator (data_processing_fl36.py in this repository, md5 46e0962a..., identical in all 40
OD_Image_* repositories of the detection series) is
imported for everything except make_image: sources, class pools, constants, seeds, XML writer, JPEG settings.
make_image_with_masks() below is make_image() line for line with the same random draws in the same order,
and additionally records where each glyph went. Images and XML are therefore identical to FashionLetters36
(checked against FashionLetters36 during development), and the detection results stay directly comparable.

Mask definition (FL36-Seg v1): glyph pixels with resized intensity > 30 - the same threshold that defines the
FL36 boxes, so each mask's tight box IS its XML box. Objects never overlap (non-overlapping boxes with a 4 px
gap), so masks are disjoint and there is no occlusion. Faint anti-aliased edge pixels (intensity 1..30) are
background in both the boxes and the masks.

Output: $PHYSICAL_AI_ROOT/datasets/FashionLetters36_Seg/{JPEGImages, Annotations, SegmentationObject,
        SegmentationClass, ImageSets/Main, ImageSets/Segmentation}   (VOC layout, palette PNGs)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEV3 = Path(__file__).resolve().parent
sys.path.insert(0, str(DEV3))
from common import config as C  # noqa: E402

GENERATOR_CANDIDATES = [DEV3 / "data_processing_fl36.py"]
MASK_THR = 30                    # = the box threshold of the original generator (g > 30)


def load_generator(path=None):
    for p in ([Path(path).expanduser()] if path else GENERATOR_CANDIDATES):
        if p.is_file():
            spec = importlib.util.spec_from_file_location("data_processing_fl36", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, p
    raise FileNotFoundError("data_processing_fl36.py not found; pass --generator")


def make_image_with_masks(G, rng, pools):
    """G.make_image + masks. Keep the random draws EXACTLY as in G.make_image."""
    canvas = np.zeros((G.SIZE, G.SIZE), np.uint8)
    inst = np.zeros((G.SIZE, G.SIZE), np.uint8)
    sem = np.zeros((G.SIZE, G.SIZE), np.uint8)
    boxes, n = [], int(rng.integers(G.MIN_OBJ, G.MAX_OBJ + 1))
    for _ in range(n):
        c = int(rng.integers(0, len(G.CLASSES)))
        glyph = pools[c][int(rng.integers(0, len(pools[c])))]
        s = int(rng.integers(G.MIN_PX, G.MAX_PX + 1))
        g = cv2.resize(glyph, (s, s), interpolation=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(g > 30)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        for _ in range(50):
            px, py = int(rng.integers(0, G.SIZE - s + 1)), int(rng.integers(0, G.SIZE - s + 1))
            b = (px + x0, py + y0, px + x1, py + y1)
            if all(b[2] + G.GAP <= o[0] or o[2] + G.GAP <= b[0] or b[3] + G.GAP <= o[1] or o[3] + G.GAP <= b[1]
                   for o in boxes):
                canvas[py:py + s, px:px + s] = np.maximum(canvas[py:py + s, px:px + s], g)
                boxes.append((*b, c))
                on = g > MASK_THR                                   # [seg] mask of this glyph
                inst[py:py + s, px:px + s][on] = len(boxes)         # k = k-th XML object (1-based)
                sem[py:py + s, px:px + s][on] = c + 1
                break
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), boxes, inst, sem


def voc_palette():
    pal = []
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal += [r, g, b]
    return pal


PALETTE = voc_palette()   # VOC colour map (from tools/fl36seg/seg_writer.py)


def save_index_png(arr, path, palette):
    im = Image.fromarray(arr, mode="P")
    im.putpalette(palette)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emnist-zip", default=None, help="EMNIST gzip.zip (default: datasets/downloads/emnist_gzip.zip)")
    ap.add_argument("--cache", default=str(C.DATASETS_ROOT / "downloads"))
    ap.add_argument("--generator", default=None, help="path of data_processing_fl36.py")
    ap.add_argument("--out", default=str(C.FL36SEG_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="first N images per split (quick check)")
    a = ap.parse_args()
    G, gpath = load_generator(a.generator)
    print(f"Generator    : {gpath}")
    root, cache = Path(a.out).expanduser(), Path(a.cache).expanduser()
    zip_path = Path(a.emnist_zip).expanduser() if a.emnist_zip else G.download(G.EMNIST_URL, cache / "emnist_gzip.zip")
    sources = (G.load_fashion(cache), G.load_letters(zip_path))
    for d in ("JPEGImages", "Annotations", "SegmentationObject", "SegmentationClass", "ImageSets/Main",
              "ImageSets/Segmentation"):
        (root / d).mkdir(parents=True, exist_ok=True)
    ids = {"trainval": [f"train_{i:05d}" for i in range(G.N_TRAIN)], "test": [f"test_{i:05d}" for i in range(G.N_TEST)]}
    for split, src, seed in (("trainval", "train", G.SEED), ("test", "test", G.SEED + 1)):
        rng, pools = np.random.default_rng(seed), G.by_class(sources, src)
        names = ids[split][: a.limit] if a.limit else ids[split]
        n_obj = 0
        for k, name in enumerate(names):
            img, boxes, inst, sem = make_image_with_masks(G, rng, pools)
            cv2.imwrite(str(root / "JPEGImages" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            G.write_xml(root / "Annotations" / f"{name}.xml", name, boxes)
            save_index_png(inst, root / "SegmentationObject" / f"{name}.png", PALETTE)
            save_index_png(sem, root / "SegmentationClass" / f"{name}.png", PALETTE)
            n_obj += len(boxes)
            if (k + 1) % 512 == 0:
                print(f"\r{split:9s}: {k + 1}/{len(names)}", end="", flush=True)
        for sub in ("Main", "Segmentation"):
            (root / "ImageSets" / sub / f"{split}.txt").write_text("\n".join(names) + "\n")
        print(f"\r{split:9s}: {len(names)} images, {n_obj} objects")
    print(f"FL36-Seg -> {root}\nnext: python train_<task>_<version>.py")


if __name__ == "__main__":
    main()
