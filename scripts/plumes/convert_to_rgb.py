"""Convert the selected 6,000 plume GeoTIFFs to lossless 8-bit RGB PNGs with the FROZEN conversion (run once).

NEEDS rasterio (not a project dependency); run in a throwaway environment:

    %TEMP%\\venv\\Scripts\\python -I scripts/plumes/convert_to_rgb.py --plumes datasets/IndustrialSmokePlumes/raw/images ^
        --index data_index/industrial_plumes.csv --params datasets/IndustrialSmokePlumes/rgb_preview/conversion_candidates.json ^
        --version per_channel --out-root <dir that contains datasets/>

The PNG of every indexed image is written to <out-root>/<path column of the index>, i.e.
datasets/IndustrialSmokePlumes/rgb/<positive|negative>/<name>.png, so the existing loader (`PIL.Image.open(...).convert("RGB")`)
and transforms are reused unchanged. The parameters (gain, gamma, per-channel k, shoulder y0) are read from the JSON written by
choose_rgb_conversion.py and copied, with the SHA-256 of every output file, into <out-root>/datasets/IndustrialSmokePlumes/rgb/
conversion.json. The script refuses to write over existing files: the conversion is frozen after approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from choose_rgb_conversion import convert, convert_per_channel, raw_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plumes", type=Path, required=True, help="raw images folder (positive/, negative/)")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True, help="conversion_candidates.json")
    parser.add_argument("--version", choices=["base", "per_channel"], required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N index rows (tests only)")
    args = parser.parse_args()

    params = json.loads(args.params.read_text(encoding="utf-8"))[args.version]
    with args.index.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[: args.limit]
    out_dir = args.out_root / "datasets" / "IndustrialSmokePlumes" / "rgb"
    digests = {}
    for n, row in enumerate(rows, 1):
        target = args.out_root / row["path"]
        if target.exists():
            sys.exit(f"{target} exists; the conversion is written once and never overwritten.")
        with rasterio.open(raw_path(args.plumes, row)) as src:
            array = src.read()
        if array.shape != (13, 120, 120):
            sys.exit(f"{raw_path(args.plumes, row)} has shape {array.shape}")
        rgb = (convert(array, params["gain"], params["gamma"]) if args.version == "base"
               else convert_per_channel(array, params["gain"], params["gamma"], tuple(params["k"]), params["y0"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(target, format="PNG")
        digests[row["path"]] = hashlib.sha256(target.read_bytes()).hexdigest()
        if n % 1000 == 0:
            print(f"{n}/{len(rows)}", flush=True)
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "conversion.json").write_text(json.dumps(
        {"version": args.version, "parameters": params, "images": len(rows), "index": str(args.index),
         "sha256": digests}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} PNGs under {out_dir}")


if __name__ == "__main__":
    main()
