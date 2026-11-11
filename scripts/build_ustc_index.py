"""Build data_index/ustc_smokers.csv (+ .meta.json) from the extracted USTC_SmokeRS folder, and report the image format.

    python scripts/build_ustc_index.py --dataset-root <dir that contains datasets/>

The class of an image is the name of the folder it sits in (smoke, cloud, dust, haze, land, seaside; case-insensitive,
at any depth beneath datasets/USTC_SmokeRS/). Images outside such a folder are listed and make the script fail, because
a silently skipped image would change the per-class counts the protocol checks against the paper (6,225 in total).
It also prints the numbers for the "Image format check" table in external/USTC_PROTOCOL.md (bit depth, channels).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bootstrap import setup_path

ROOT = setup_path()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from firecls.data.ustc import IMAGE_SUFFIXES, USTC_CLASSES, USTC_DIR, USTC_INDEX, binary_label  # noqa: E402
from firecls.deployment.artifacts import write_json  # noqa: E402

EXPECTED_TOTAL = 6225  # Ba et al. (2019)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT, help="directory that contains datasets/")
    parser.add_argument("--index-root", type=Path, default=ROOT, help="directory that contains data_index/")
    parser.add_argument("--format-sample", type=int, default=200, help="images per class opened for the format check")
    return parser.parse_args()


def class_of(path: Path, base: Path) -> str | None:
    folders = [part.lower() for part in path.relative_to(base).parts[:-1]]
    hits = [name for name in USTC_CLASSES if name in folders]
    return hits[-1] if hits else None


def main() -> None:
    args = parse_args()
    base = args.dataset_root / USTC_DIR
    if not base.is_dir():
        sys.exit(f"{base} not found. Download and extract USTC_SmokeRS there first (external/README.md, Phase 1).")

    files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    rows, unplaced = [], []
    for path in files:
        label = class_of(path, base)
        if label is None:
            unplaced.append(path)
        else:
            rows.append((path, label))
    if unplaced:
        sys.exit(f"{len(unplaced)} images are not inside a class folder, e.g. {unplaced[:3]}")
    if not rows:
        sys.exit(f"No images found beneath {base}")

    counts = Counter(label for _, label in rows)
    total = len(rows)
    print("Images per class:", {name: counts.get(name, 0) for name in USTC_CLASSES}, "total", total)
    if total != EXPECTED_TOTAL:
        print(f"WARNING: total {total} != {EXPECTED_TOTAL} expected by the paper; do not use a mirror whose counts differ.")

    # Format check on a sample from every class.
    by_class = defaultdict(list)
    for path, label in rows:
        by_class[label].append(path)
    modes, sizes, dtypes, maxima, unreadable = Counter(), Counter(), Counter(), [], []
    for label, paths in by_class.items():
        for path in paths[:: max(1, len(paths) // args.format_sample)]:
            try:
                with Image.open(path) as image:
                    modes[image.mode] += 1
                    sizes[image.size] += 1
                    array = np.asarray(image)
                    dtypes[str(array.dtype)] += 1
                    maxima.append(int(array.max()))
                    image.convert("RGB").load()
            except Exception as error:  # noqa: BLE001 - report every unreadable file
                unreadable.append((str(path), repr(error)))
    print("PIL modes:", dict(modes), "| sizes:", dict(sizes), "| dtypes:", dict(dtypes))
    if maxima:
        print(f"max pixel value over the sample: min {min(maxima)}, max {max(maxima)} (255 = 8-bit range)")
    if unreadable:
        sys.exit(f"{len(unreadable)} sampled images failed to open/convert: {unreadable[:3]}")

    index_path = args.index_root / USTC_INDEX
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "label6", "label_bin", "split"])
        for path, label in rows:
            writer.writerow([path.relative_to(args.dataset_root).as_posix(), label, binary_label(label), "test"])
    write_json(index_path.with_suffix(".meta.json"), {
        "dataset": "USTC_SmokeRS", "classes6": USTC_CLASSES, "positive_class": "smoke",
        "counts": {name: counts.get(name, 0) for name in USTC_CLASSES}, "total": total,
        "expected_total": EXPECTED_TOTAL, "split": "all images are test",
        "format_sample": {"modes": dict(modes), "sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
                          "dtypes": dict(dtypes)},
        "created": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Wrote {index_path} ({total} rows)")


if __name__ == "__main__":
    main()
