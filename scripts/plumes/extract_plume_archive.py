"""Extract the Industrial Smoke Plumes archive (Mommert et al., 2020) with Windows-safe file names (run once).

    python -I scripts/plumes/extract_plume_archive.py --archive D:/Thesis-Project/images.tar.gz ^
        --dest datasets/IndustrialSmokePlumes/raw/images

The archive members are `positive/<name>.tif` (3,750) and `negative/<name>.tif` (17,600), with names like
`10003_2019-01-21T10:56:41.330Z_0.tif`. **The folder is the only record of the label (plume or no plume), so it is kept**:
files are written to `<dest>/positive/` and `<dest>/negative/`. The colons are not allowed in Windows file names, so every
':' in a name is replaced by '_' on disk. Nothing else changes, and `manifest.csv` (written into --dest) keeps the
original and local name, the label folder, the size and the SHA-256 of every file, so the mapping is reversible.

Safety: the archive is untrusted data. Only regular files directly inside `positive/` or `negative/` are written (any other
path stops the script; no links, no `..`), and the script refuses a non-empty --dest or two members that map to one path.
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath


LABEL_FOLDERS = ("positive", "negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True, help="must be empty or absent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dest.exists() and any(args.dest.iterdir()):
        sys.exit(f"{args.dest} is not empty; refusing to overwrite.")
    args.dest.mkdir(parents=True, exist_ok=True)

    rows, seen, skipped = [], set(), 0
    with tarfile.open(args.archive, mode="r|gz") as archive:  # streamed: no random access needed
        for member in archive:
            if not member.isreg():
                skipped += 1
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 2 or parts[0] not in LABEL_FOLDERS or parts[1] in (".", ".."):
                sys.exit(f"Unexpected member path {member.name!r}; expected positive/<file> or negative/<file>")
            label, original = parts
            local = original.replace(":", "_")
            if (label, local) in seen:
                sys.exit(f"Two members map to the same path {label}/{local!r}")
            seen.add((label, local))
            (args.dest / label).mkdir(exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with (args.dest / label / local).open("wb") as out, archive.extractfile(member) as source:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != member.size:
                sys.exit(f"{original}: wrote {size} bytes, archive says {member.size}")
            rows.append((label, original, local, size, digest.hexdigest()))
            if len(rows) % 2000 == 0:
                print(f"{len(rows)} files", flush=True)

    with (args.dest / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["label_src", "original_name", "local_name", "size", "sha256"])
        writer.writerows(sorted(rows))
    print(f"OK: {len(rows)} files written to {args.dest} ({skipped} non-file members skipped); manifest.csv written.")


if __name__ == "__main__":
    main()
