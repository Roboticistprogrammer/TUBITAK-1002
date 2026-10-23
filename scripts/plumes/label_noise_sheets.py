"""Blind contact sheets for the label-noise estimate (external/PLUME_PROTOCOL.md, Section 6).

    python -I scripts/plumes/label_noise_sheets.py --index data_index/industrial_plumes.csv --rgb-root <dir with datasets/> ^
        --segmentation datasets/IndustrialSmokePlumes/raw/segmentation_labels.tar.gz --out-dir <folder>

A fixed-seed (0) random sample of 100 plume and 100 non-plume TRAINING images is shuffled together and drawn on sheets of 20
(5 x 4) at their full field of view, labelled only with a running number, so the reviewer does not see the source label. The
key (number -> path, label, whether the image has a manual segmentation polygon in the Zenodo labels) is written to key.csv and
must not be opened until every number has been judged. Nothing is relabelled or removed; the output is an estimate of the rate
of plume images without a visible plume and of non-plume images with a visible plume.
"""
from __future__ import annotations

import argparse
import csv
import re
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE, COLS, ROWS = 192, 5, 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    with args.index.open(encoding="utf-8") as stream:
        train = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    plume = [r for r in train if r["label"] == "smoke"]
    plain = [r for r in train if r["label"] == "neither"]
    sample = [plume[i] for i in rng.choice(len(plume), args.per_class, replace=False)]
    sample += [plain[i] for i in rng.choice(len(plain), args.per_class, replace=False)]
    order = rng.permutation(len(sample))
    sample = [sample[i] for i in order]

    with tarfile.open(args.segmentation) as archive:  # names look like <image name>_features.json (':' kept in the archive)
        polygons = {re.sub(r"_features\.json$", "", m.name).replace(":", "_") for m in archive.getmembers() if m.isfile()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = COLS * ROWS
    with (args.out_dir / "key.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["number", "path", "label", "label_src", "has_polygon"])
        for n, row in enumerate(sample, 1):
            stem = Path(row["path"]).stem
            writer.writerow([n, row["path"], row["label"], row["label_src"], int(stem in polygons)])
    for s in range((len(sample) + per_sheet - 1) // per_sheet):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(sample[s * per_sheet:(s + 1) * per_sheet]):
            n = s * per_sheet + j + 1
            image = Image.open(args.rgb_root / row["path"]).convert("RGB").resize((TILE, TILE), Image.BILINEAR)
            x, y = (j % COLS) * TILE, (j // COLS) * (TILE + 14)
            sheet.paste(image, (x, y + 14))
            draw.text((x + 4, y + 1), f"#{n}", fill="black")
        sheet.save(args.out_dir / f"sheet_{s + 1:02d}.png")
    print(f"{len(sample)} images, {(len(sample) + per_sheet - 1) // per_sheet} sheets in {args.out_dir}; key in key.csv (do not open yet)")


if __name__ == "__main__":
    main()
