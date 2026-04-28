from __future__ import annotations

import argparse
from collections import Counter
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from bootstrap import setup_path

setup_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image size statistics for a dataset folder.")
    parser.add_argument("--root", type=Path, required=True, help="Root folder containing images")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=5,
        help="Maximum number of images to scan per category folder (default: 5)",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"],
        help="Extensions to include",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exts = {ext.lower() for ext in args.exts}

    files_by_category = {}
    for path in args.root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        relative_parts = path.relative_to(args.root).parts
        category = relative_parts[0] if len(relative_parts) > 1 else args.root.name
        files_by_category.setdefault(category, []).append(path)

    rng = random.Random(42)
    files = []
    for category in sorted(files_by_category):
        category_files = files_by_category[category]
        rng.shuffle(category_files)
        files.extend(category_files[: args.per_category_limit])

    if args.limit:
        files = files[: args.limit]

    if not files:
        print("No images found.")
        return

    print(f"Categories found: {len(files_by_category)}")
    print(f"Images sampled per category: {args.per_category_limit}")

    widths = []
    heights = []
    size_counter = Counter()

    for path in tqdm(files, desc="Scanning"):
        with Image.open(path) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)
        size_counter[(w, h)] += 1

    def summarize(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    w_stats = summarize(widths)
    h_stats = summarize(heights)

    print(f"Images scanned: {len(files)}")
    print(f"Width  min/max/mean: {w_stats['min']} / {w_stats['max']} / {w_stats['mean']:.2f}")
    print(f"Height min/max/mean: {h_stats['min']} / {h_stats['max']} / {h_stats['mean']:.2f}")
    print("Most common sizes:")
    for (w, h), count in size_counter.most_common(10):
        print(f"  {w}x{h}: {count}")


if __name__ == "__main__":
    main()
