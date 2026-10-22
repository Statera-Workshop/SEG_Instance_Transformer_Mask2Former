"""
Step 1 (algorithm validation track) - build the FashionLetters36 detection dataset.

    python data_processing_fl36.py                      # download sources, generate, verify
    python data_processing_fl36.py --emnist-zip ~/Downloads/gzip.zip

36 classes = Fashion-MNIST (10) + EMNIST letters (26, upper and lower case merged).
608 x 608 RGB canvases, 3-10 objects per image, random size 32-160 px, no overlap, black background.
    train : 4,096 images, glyphs from the source TRAIN splits
    test  : 1,024 images, glyphs from the source TEST splits   (no glyph is shared)
Fixed seed (777): every user generates exactly the same dataset.

Output (VOC layout, read by the unchanged Image pipeline):
    $PHYSICAL_AI_ROOT/datasets/FashionLetters36/{JPEGImages, Annotations, ImageSets/Main/{trainval,test}.txt}
Train and evaluate on it with  python train_image_<version>.py

Purpose: check that each algorithm is implemented correctly (it must converge quickly on this
easy, fully controlled data) - separate from benchmarking on VOC / COCO.

Sources:
    Fashion-MNIST  https://github.com/zalandoresearch/fashion-mnist (MIT)
    EMNIST         https://www.nist.gov/itl/products-and-services/emnist-dataset  (gzip.zip, ~0.56 GB)
"""

import argparse
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config as C

FASHION_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
FASHION_NAMES = ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
LETTERS = [chr(ord("A") + i) for i in range(26)]
CLASSES = FASHION_NAMES + LETTERS                      # = common.config.FL36_CLASSES
SIZE, N_TRAIN, N_TEST, SEED = 608, 4096, 1024, 777
MIN_OBJ, MAX_OBJ, MIN_PX, MAX_PX, GAP = 3, 10, 32, 160, 4


def download(url, dst):
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Download     : {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r               {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dst)
    return dst


def idx(data):
    """Parse an IDX file (bytes) -> numpy array."""
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_fashion(cache):
    out = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        img = idx(gzip.open(download(FASHION_URL + f"{prefix}-images-idx3-ubyte.gz", cache / f"fashion_{prefix}_images.gz")).read())
        lab = idx(gzip.open(download(FASHION_URL + f"{prefix}-labels-idx1-ubyte.gz", cache / f"fashion_{prefix}_labels.gz")).read())
        out[split] = (img, lab.astype(np.int64))                       # labels 0..9
    return out


def load_letters(zip_path):
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for split in ("train", "test"):
            img = idx(gzip.open(io.BytesIO(z.read(names[f"emnist-letters-{split}-images-idx3-ubyte.gz"]))).read())
            lab = idx(gzip.open(io.BytesIO(z.read(names[f"emnist-letters-{split}-labels-idx1-ubyte.gz"]))).read())
            img = img.transpose(0, 2, 1)                                # EMNIST is stored transposed
            out[split] = (img, lab.astype(np.int64) - 1 + 10)           # letters 1..26 -> classes 10..35
    return out


def by_class(sources, split):
    pools = {}
    for img, lab in (sources[0][split], sources[1][split]):
        for c in np.unique(lab):
            pools[int(c)] = img[lab == c]
    return pools


def make_image(rng, pools):
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    boxes, n = [], int(rng.integers(MIN_OBJ, MAX_OBJ + 1))
    for _ in range(n):
        c = int(rng.integers(0, len(CLASSES)))
        glyph = pools[c][int(rng.integers(0, len(pools[c])))]
        s = int(rng.integers(MIN_PX, MAX_PX + 1))
        g = cv2.resize(glyph, (s, s), interpolation=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(g > 30)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1      # tight box inside the patch
        for _ in range(50):                                                    # non-overlapping placement
            px, py = int(rng.integers(0, SIZE - s + 1)), int(rng.integers(0, SIZE - s + 1))
            b = (px + x0, py + y0, px + x1, py + y1)
            if all(b[2] + GAP <= o[0] or o[2] + GAP <= b[0] or b[3] + GAP <= o[1] or o[3] + GAP <= b[1] for o in boxes):
                canvas[py:py + s, px:px + s] = np.maximum(canvas[py:py + s, px:px + s], g)
                boxes.append((*b, c))
                break
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), boxes


def write_xml(path, name, boxes):
    objs = "".join(
        f"<object><name>{CLASSES[c]}</name><difficult>0</difficult><bndbox><xmin>{x0 + 1}</xmin><ymin>{y0 + 1}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>" for x0, y0, x1, y1, c in boxes)
    path.write_text(f"<annotation><filename>{name}.jpg</filename><size><width>{SIZE}</width><height>{SIZE}</height>"
                    f"<depth>3</depth></size>{objs}</annotation>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emnist-zip", default=None, help="EMNIST gzip.zip (downloaded from NIST if omitted)")
    ap.add_argument("--cache", default=str(C.DATASETS_ROOT / "downloads"))
    a = ap.parse_args()
    root, cache = C.DATASETS_ROOT / "FashionLetters36", Path(a.cache).expanduser()
    ids = {"trainval": [f"train_{i:05d}" for i in range(N_TRAIN)], "test": [f"test_{i:05d}" for i in range(N_TEST)]}
    if all((root / "JPEGImages" / f"{i}.jpg").is_file() for s in ids.values() for i in s):
        print(f"FashionLetters36: complete ({N_TRAIN} train / {N_TEST} test) - nothing to do")
        return
    zip_path = Path(a.emnist_zip).expanduser() if a.emnist_zip else download(EMNIST_URL, cache / "emnist_gzip.zip")
    sources = (load_fashion(cache), load_letters(zip_path))
    for d in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (root / d).mkdir(parents=True, exist_ok=True)
    counts = np.zeros(len(CLASSES), int)
    for split, src, seed in (("trainval", "train", SEED), ("test", "test", SEED + 1)):
        rng, pools = np.random.default_rng(seed), by_class(sources, src)
        for k, name in enumerate(ids[split]):
            img, boxes = make_image(rng, pools)
            cv2.imwrite(str(root / "JPEGImages" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_xml(root / "Annotations" / f"{name}.xml", name, boxes)
            if split == "trainval":
                for b in boxes:
                    counts[b[4]] += 1
            if (k + 1) % 512 == 0:
                print(f"\r{split:9s}: {k + 1}/{len(ids[split])}", end="", flush=True)
        (root / "ImageSets/Main" / f"{split}.txt").write_text("\n".join(ids[split]) + "\n")
        print()
    print(f"FashionLetters36: {N_TRAIN} train / {N_TEST} test images -> {root}")
    print(f"train objects per class: min {counts.min()}  max {counts.max()}  total {counts.sum()}")
    print("Data processing: done - next: python train_image_<version>.py")


if __name__ == "__main__":
    main()
