from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import setup_path

ROOT = setup_path()

from firecls.config import DEFAULT_CLASSES, get_dataset_specs
from firecls.data.tdml import create_index_csv
from firecls.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create image-level CSV from TDML annotations.")
    parser.add_argument("--dataset", choices=["cv", "rs", "uav"], help="Dataset shortcut name")
    parser.add_argument("--tdml-json", type=Path, help="Path to TDML JSON file")
    parser.add_argument("--images-root", type=Path, help="Path to images root")
    parser.add_argument("--output-csv", type=Path, default=Path("data_index/tdml_index.csv"))
    parser.add_argument("--both-as", choices=["both", "fire", "smoke"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset:
        specs = get_dataset_specs()
        spec = specs[args.dataset]
        tdml_json = spec.tdml_json
        images_root = spec.images_root
        output_csv = Path(f"data_index/{spec.name.lower()}_tdml.csv")
    else:
        if not args.tdml_json or not args.images_root:
            raise SystemExit("Provide --tdml-json and --images-root if --dataset is not set.")
        tdml_json = args.tdml_json
        images_root = args.images_root
        output_csv = args.output_csv

    counts = create_index_csv(tdml_json, images_root, output_csv, both_as=args.both_as, limit=args.limit)
    save_json(output_csv.with_suffix(".meta.json"), {
        "tdml_json": str(tdml_json),
        "images_root": str(images_root),
        "classes": DEFAULT_CLASSES,
        "both_as": args.both_as,
        "counts": counts,
    })

    print(f"✓ Wrote CSV: {output_csv}")
    print(f"✓ Metadata: {output_csv.with_suffix('.meta.json')}")
    print(counts)


if __name__ == "__main__":
    main()
