"""Inspect the extracted Industrial Smoke Plumes images: names, label coding, sites and GeoTIFF format.

NEEDS rasterio (not a project dependency); run it in a throwaway environment, never inside the data folder:

    %TEMP%\\venv\\Scripts\\python -I scripts/plumes/inspect_plumes.py --images datasets/IndustrialSmokePlumes/raw/images

Prints everything it finds and writes nothing. The file name is `<site>_<UTC timestamp with ':' replaced by '_'>_<n>.tif`.
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

import numpy as np
import rasterio

PATTERN = re.compile(r"^(?P<site>\d+)_(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>[\d_.]+)Z_(?P<n>\d+)\.tif$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--value-sample", type=int, default=400, help="images read in full for value statistics")
    args = parser.parse_args()

    files = sorted(args.images.glob("*.tif"))
    print("tif files:", len(files))
    unmatched = [p.name for p in files if not PATTERN.match(p.name)]
    print("names not matching the pattern:", len(unmatched), unmatched[:5])
    parts = [PATTERN.match(p.name) for p in files if PATTERN.match(p.name)]
    suffix = collections.Counter(m["n"] for m in parts)
    print("trailing number counts (top 12):", suffix.most_common(12), "| distinct:", len(suffix))
    sites = collections.Counter(m["site"] for m in parts)
    print("distinct sites:", len(sites), "| images per site: min", min(sites.values()), "median",
          int(np.median(list(sites.values()))), "max", max(sites.values()))
    years = collections.Counter(m["date"][:4] for m in parts)
    print("images per year:", dict(sorted(years.items())))

    profiles = collections.Counter()
    shapes = collections.Counter()
    for p in files:
        with rasterio.open(p) as src:
            profiles[(src.count, tuple(src.dtypes), src.crs.to_string() if src.crs else None, src.nodata)] += 1
            shapes[(src.height, src.width)] += 1
    print("(bands, dtypes, crs, nodata) groups:", [(k[0], set(k[1]), k[2], k[3], n) for k, n in profiles.most_common(5)])
    print("shapes:", dict(shapes.most_common(5)))

    with rasterio.open(files[0]) as src:
        print("band descriptions:", src.descriptions)
        print("tags:", src.tags(), "| band-1 tags:", src.tags(1))
        print("transform:", src.transform, "| pixel size:", src.res)

    step = max(1, len(files) // args.value_sample)
    sample = [files[i] for i in range(0, len(files), step)]
    stack = np.stack([rasterio.open(p).read().astype(np.float64) for p in sample])  # (n, bands, h, w)
    print(f"value statistics over {len(sample)} images, per band (min / p1 / median / p99 / max):")
    for b in range(stack.shape[1]):
        v = stack[:, b].ravel()
        print(f"  band {b + 1:2d}: {v.min():8.0f} {np.percentile(v, 1):8.0f} {np.median(v):8.0f} "
              f"{np.percentile(v, 99):8.0f} {v.max():8.0f}")


if __name__ == "__main__":
    main()
