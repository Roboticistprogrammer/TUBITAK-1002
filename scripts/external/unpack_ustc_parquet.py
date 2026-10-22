"""Unpack the Hugging Face copy of USTC_SmokeRS into datasets/USTC_SmokeRS/<class>/<file>.tif (run once).

NEEDS pyarrow, which is deliberately NOT in requirements.txt or any other requirements file. Run this script once in a
throwaway environment; nothing else in the pipeline may import pyarrow:

    python -m venv %TEMP%\\ustc_venv
    %TEMP%\\ustc_venv\\Scripts\\python -m pip install pyarrow
    %TEMP%\\ustc_venv\\Scripts\\python -I scripts/external/unpack_ustc_parquet.py ^
        --source datasets/USTC_SmokeRS_hf --dest datasets/USTC_SmokeRS

Source: https://huggingface.co/datasets/jonathan-roberts1/USTC_SmokeRS at the revision recorded in
external/USTC_PROTOCOL.md (download the three data/*.parquet files from that revision into --source).

What it does, and why:
* The parquet stores each image as the original encoded TIFF (`image.bytes`) plus the original file name
  (`image.path`, e.g. cloud_1.tif). The bytes are written to disk unchanged (no decode, no re-encode), under the
  original file name, so every prediction can be traced to an image and the evaluation uses the same decode path as
  FASDD-RS (`PIL.Image.open(...).convert("RGB")`) from a real .tif file.
* The class comes from the class NAME in the file name, never from the integer `label`. The integer is only
  cross-checked: the Hugging Face card order (0 cloud, 1 dust, 2 haze, 3 land, 4 seaside, 5 smoke) differs from the
  protocol's order, and the script fails if any file's name and integer disagree with that card order.
* It refuses to write into a non-empty --dest, then verifies: total count, per-class counts against Table 2 of Ba et
  al. (2019), and the SHA-256 of EVERY file re-read from disk against the parquet bytes. It also writes
  manifest.csv (file, class, hf_label, sha256) next to the class folders.
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# Hugging Face dataset card order (dataset_info.features.label.names). Used only as a cross-check.
HF_LABEL_NAMES = {0: "cloud", 1: "dust", 2: "haze", 3: "land", 4: "seaside", 5: "smoke"}
# Table 2 of Ba et al. (2019), Remote Sensing 11(14):1702: images per class (total 6,225).
PAPER_COUNTS = {"cloud": 1164, "dust": 1009, "haze": 1002, "land": 1027, "seaside": 1007, "smoke": 1016}
NAME_PATTERN = re.compile(r"^(cloud|dust|haze|land|seaside|smoke)_(\d+)\.tif$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="folder with data/*.parquet")
    parser.add_argument("--dest", type=Path, required=True, help="e.g. datasets/USTC_SmokeRS (must be empty or absent)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted((args.source / "data").glob("*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {args.source / 'data'}")
    if args.dest.exists() and any(args.dest.iterdir()):
        sys.exit(f"{args.dest} is not empty; refusing to overwrite. Remove it yourself if you really want to redo this.")

    written: dict[str, tuple[str, int, str]] = {}  # file name -> (class, hf label, sha256 of the parquet bytes)
    for parquet in files:
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128):
            columns = batch.to_pydict()
            for image, label in zip(columns["image"], columns["label"]):
                match = NAME_PATTERN.match(image["path"])
                if not match:
                    sys.exit(f"Unexpected file name {image['path']!r}")
                name, class_name = image["path"], match.group(1)
                if HF_LABEL_NAMES[label] != class_name:
                    sys.exit(f"{name}: file name says {class_name} but hf label {label} is {HF_LABEL_NAMES[label]}")
                if name in written:
                    sys.exit(f"Duplicate file name {name}")
                target = args.dest / class_name / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image["bytes"])  # original bytes, unchanged
                written[name] = (class_name, label, hashlib.sha256(image["bytes"]).hexdigest())

    # Verification, all from the files on disk.
    problems = []
    counts = Counter(class_name for class_name, _, _ in written.values())
    if len(written) != sum(PAPER_COUNTS.values()):
        problems.append(f"total {len(written)} != {sum(PAPER_COUNTS.values())}")
    if dict(counts) != PAPER_COUNTS:
        problems.append(f"per-class counts {dict(sorted(counts.items()))} != paper Table 2 {PAPER_COUNTS}")
    for name, (class_name, _, expected) in written.items():
        actual = hashlib.sha256((args.dest / class_name / name).read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{name}: sha256 on disk differs from the parquet bytes")
    on_disk = sum(1 for p in args.dest.rglob("*.tif"))
    if on_disk != len(written):
        problems.append(f"{on_disk} .tif files on disk, {len(written)} written")

    with (args.dest / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["file", "class", "hf_label", "sha256"])
        for name in sorted(written):
            class_name, label, digest = written[name]
            writer.writerow([name, class_name, label, digest])
    manifest_sha = hashlib.sha256((args.dest / "manifest.csv").read_bytes()).hexdigest()

    print("per-class counts:", dict(sorted(counts.items())), "total", len(written))
    print("manifest.csv sha256:", manifest_sha)
    if problems:
        sys.exit("VERIFICATION FAILED:\n  " + "\n  ".join(problems[:20]))
    print(f"OK: {len(written)} files written, per-class counts match Table 2, "
          f"all {len(written)} SHA-256 re-read from disk match the parquet bytes.")


if __name__ == "__main__":
    main()
