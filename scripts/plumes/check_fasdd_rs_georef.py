"""Georeferencing of the FASDD-RS images in the index (external/PLUME_PROTOCOL.md, Section 5).

NEEDS rasterio (not a project dependency); run in a throwaway environment:

    %TEMP%\\venv\\Scripts\\python -I scripts/plumes/check_fasdd_rs_georef.py --index data_index/fasdd_rs_tdml.csv ^
        --images datasets/Fire_Remote_Sensing/images

Reports, for the images listed in the index (not for every file in the folder): how many have no CRS per split, how many are not
1000 x 1000 px, and the pixel size in metres (north-south, and east-west at the latitude of the image; the grid is square in
degrees, so the east-west size shrinks with latitude).
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path

import numpy as np
import rasterio

METRES_PER_DEGREE = 111_320.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    with args.index.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print("indexed rows", len(rows), dict(collections.Counter(r["split"] for r in rows)))
    no_crs, odd_size, north_south, east_west, missing = collections.Counter(), [], set(), [], 0
    for row in rows:
        name = Path(row["path"].replace("\\", "/")).name
        path = args.images / name
        if not path.exists():
            missing += 1
            continue
        with rasterio.open(path) as src:
            if (src.height, src.width) != (1000, 1000):
                odd_size.append((row["split"], name, src.height, src.width))
            if src.crs is None or src.transform.is_identity:
                no_crs[row["split"]] += 1
                continue
            latitude = (src.bounds.top + src.bounds.bottom) / 2
            north_south.add(round(src.res[1] * METRES_PER_DEGREE, 3))
            east_west.append(src.res[0] * METRES_PER_DEGREE * math.cos(math.radians(latitude)))
    east_west = np.array(east_west)
    print(f"missing files: {missing} | indexed images WITHOUT a CRS: {sum(no_crs.values())} {dict(no_crs)}")
    print(f"indexed images not 1000 x 1000: {len(odd_size)} {odd_size}")
    print("north-south pixel (m) among georeferenced indexed images:", sorted(north_south))
    print(f"east-west pixel (m): n={len(east_west)} min {east_west.min():.2f} median {np.median(east_west):.2f} max {east_west.max():.2f}")


if __name__ == "__main__":
    main()
