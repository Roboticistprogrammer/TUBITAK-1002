"""Check that plume and non-plume images are equally sharp (external/PLUME_PROTOCOL.md, Section 6).

    python -I scripts/plumes/check_sharpness.py --index data_index/industrial_plumes.csv --rgb-root <dir that contains datasets/>

Reads the frozen converted PNGs listed in the index (no rasterio needed). Both classes are enlarged from 120 px in the same way,
so upsampling blur cannot be a label cue if the sharpness of the two classes is similar. For every selected image the script takes
the luminance, resizes to 224 px and centre-crops 192 px (the evaluation view), and computes the variance of the 4-neighbour
Laplacian. It prints mean, standard deviation and median per label and split, and the difference of the means in units of the
pooled standard deviation (Cohen's d).
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def laplacian_variance(gray: np.ndarray) -> float:
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    args = parser.parse_args()

    values = collections.defaultdict(list)
    with args.index.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rgb = np.asarray(Image.open(args.rgb_root / row["path"]).convert("RGB"), dtype=np.float32)
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            view = Image.fromarray(gray, mode="F").resize((224, 224), Image.BILINEAR)
            left = (224 - 192) // 2
            crop = np.asarray(view.crop((left, left, left + 192, left + 192)))
            sharp = laplacian_variance(crop)
            values[(row["label"], row["split"])].append(sharp)
            values[(row["label"], "all")].append(sharp)
    print("Laplacian variance of the 192 x 192 evaluation view (higher = sharper):")
    for split in ("train", "val", "test", "all"):
        stats = {}
        for label in ("smoke", "neither"):
            v = np.array(values[(label, split)])
            stats[label] = v
            print(f"  {split:5s} {label:8s} n={len(v):5d} mean {v.mean():8.2f} sd {v.std():8.2f} median {np.median(v):8.2f}")
        pooled = np.sqrt((stats["smoke"].var() + stats["neither"].var()) / 2)
        print(f"  {split:5s} difference of means (plume - non-plume) in pooled SDs: {(stats['smoke'].mean() - stats['neither'].mean()) / pooled:+.3f}")


if __name__ == "__main__":
    main()
