"""Build the per-image table of the Industrial Smoke Plumes data (label folder, site, date, location) as a CSV.

NEEDS rasterio (not a project dependency); run in a throwaway environment:

    %TEMP%\\venv\\Scripts\\python -I scripts/plumes/site_table.py --images datasets/IndustrialSmokePlumes/raw/images ^
        --out datasets/IndustrialSmokePlumes/raw/image_table.csv

Columns: label_src (positive/negative folder), file, site (the leading number of the name), date, n (trailing number),
lon, lat (centre of the image, WGS84), height, width. Only the GeoTIFF headers are read.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import rasterio
from rasterio.warp import transform

PATTERN = re.compile(r"^(?P<site>\d+)_(?P<date>\d{4}-\d{2}-\d{2})T[\d_.]+Z_(?P<n>\d+)\.tif$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="folder with positive/ and negative/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for label in ("positive", "negative"):
        for path in sorted((args.images / label).glob("*.tif")):
            match = PATTERN.match(path.name)
            if not match:
                raise ValueError(f"Unexpected name {path.name}")
            with rasterio.open(path) as src:
                cx, cy = src.transform * (src.width / 2, src.height / 2)
                (lon,), (lat,) = transform(src.crs, "EPSG:4326", [cx], [cy])
                rows.append([label, path.name, match["site"], match["date"], match["n"], f"{lon:.5f}", f"{lat:.5f}",
                             src.height, src.width])
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["label_src", "file", "site", "date", "n", "lon", "lat", "height", "width"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
