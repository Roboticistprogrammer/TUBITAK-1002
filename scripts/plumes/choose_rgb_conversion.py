"""Choose ONE fixed 13-band -> 8-bit RGB conversion for the plume images by matching how FASDD-RS images look.

NEEDS rasterio (not a project dependency); run in a throwaway environment:

    %TEMP%\\venv\\Scripts\\python -I scripts/plumes/choose_rgb_conversion.py ^
        --plumes datasets/IndustrialSmokePlumes/raw/images --index data_index/industrial_plumes.csv ^
        --fasdd-index data_index/fasdd_rs_tdml.csv --fasdd-images datasets/Fire_Remote_Sensing/images ^
        --preview-dir datasets/IndustrialSmokePlumes/rgb_preview

Bands: the GeoTIFF band order is B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12 (band 11 = B10 comes from L1C and has
the small values of the cirrus band, which confirms the order). RGB = B4, B3, B2 = bands 4, 3, 2. Values are L2A
reflectance x 10,000 (uint16). Conversions are fixed formulas, never per-image.

1. BASE (grid search, fixed before): rgb = clip(reflectance * gain, 0, 1) ** gamma * 255, reflectance = value / 10,000. The score
   of a candidate is the mean absolute difference, over the three channels and over the percentiles 1, 5, 10, 25, 50, 75, 90,
   95, 99, between the pooled pixel distribution of 400 converted plume images (random, seed 0, from the selected 6,000) and that
   of 200 FASDD-RS **training** images, in 8-bit units. The best candidate (gain 2.0, gamma 0.6, score 15.3) is the base.
2. PER-CHANNEL version (decision of PLUME_FINAL_DECISIONS.md, Section 4), parameters fitted on 1,000 plume **train** images and the
   FASDD-RS **train** split only:
   z_c = (reflectance_c * gain) ** gamma                      (no clipping)
   y_c = k_c * z_c, with k_c chosen so that the median of channel c equals the FASDD-RS training median of channel c
   out_c = 255 * h(y_c), with the highlight shoulder h(y) = y for y <= y0 and
           h(y) = y0 + (1 - y0) * (1 - exp(-(y - y0) / (1 - y0))) above it (smooth, slope 1 at y0, never reaches 1);
   y0 is chosen so that the pooled p99 of the converted plume train pixels is 220 (FASDD-RS: 200, 212, 221 per channel), so bright
   steam and cloud are compressed, not clipped. (The decision text says "cap p99 at about 220"; this is how that was read: one
   shoulder parameter fixed by the p99 target, which leaves the medians unchanged because y0 is above them.)
   It is adopted only if its score on the SAME 400-image sample as the base is not worse than the base score.
Content differs between the datasets (industrial sites in Europe against fire and smoke scenes worldwide), so scores cannot reach
zero; they rank conversions only. Preview sheets are written for the base and per-channel versions beside FASDD-RS images.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

PCT = (1, 5, 10, 25, 50, 75, 90, 95, 99)
RGB_BANDS = (4, 3, 2)  # rasterio band numbers (1-based): B4 red, B3 green, B2 blue
GAINS = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)
GAMMAS = (1.0, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45)
P99_TARGET = 220.0


def model_view(image: Image.Image, resize: int = 224, crop: int = 192) -> Image.Image:
    """What the evaluation transform shows the model: shorter side to 224, centre crop 192 (as torchvision Resize + CenterCrop)."""
    w, h = image.size
    scale = resize / min(w, h)
    image = image.resize((max(resize, round(w * scale)), max(resize, round(h * scale))), Image.BILINEAR)
    w, h = image.size
    left, top = (w - crop) // 2, (h - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def convert(array: np.ndarray, gain: float, gamma: float) -> np.ndarray:
    """BASE conversion. array: (13, h, w) uint16 -> (h, w, 3) uint8."""
    reflectance = array[[b - 1 for b in RGB_BANDS]].astype(np.float32) / 10000.0
    scaled = np.clip(reflectance * gain, 0.0, 1.0) ** gamma
    return np.rint(scaled * 255.0).astype(np.uint8).transpose(1, 2, 0)


def shoulder(y: np.ndarray, y0: float) -> np.ndarray:
    above = np.maximum(y - y0, 0.0)
    return np.where(y <= y0, y, y0 + (1.0 - y0) * (1.0 - np.exp(-above / (1.0 - y0))))


def convert_per_channel(array: np.ndarray, gain: float, gamma: float, k: tuple, y0: float) -> np.ndarray:
    """PER-CHANNEL conversion (see the module docstring). array: (13, h, w) uint16 -> (h, w, 3) uint8."""
    reflectance = array[[b - 1 for b in RGB_BANDS]].astype(np.float32) / 10000.0
    z = (np.maximum(reflectance * gain, 0.0)) ** gamma
    y = z * np.asarray(k, dtype=np.float32)[:, None, None]
    return np.rint(np.clip(shoulder(y, y0), 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)


def distance(pixels: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(np.percentile(pixels, PCT, axis=0) - target).mean())


def fit_per_channel(train_raw: list[np.ndarray], gain: float, gamma: float, fasdd_median: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.concatenate([(np.maximum(a[[b - 1 for b in RGB_BANDS]].astype(np.float32) / 10000.0 * gain, 0.0) ** gamma)
                        .reshape(3, -1)[:, ::7] for a in train_raw], axis=1)  # (3, n)
    k = (fasdd_median / 255.0) / np.median(z, axis=1)
    lo, hi = float(np.max(fasdd_median) / 255.0) + 1e-3, 0.999  # y0 must stay above the medians so they are untouched
    y = z * k[:, None]
    for _ in range(60):  # bisection: pooled p99 of 255 * h(y) = P99_TARGET (monotone in y0)
        y0 = (lo + hi) / 2
        p99 = np.percentile(255.0 * np.clip(shoulder(y, y0), 0, 1), 99)
        lo, hi = (lo, y0) if p99 > P99_TARGET else (y0, hi)  # p99 too high -> compress more -> lower y0
    return k, (lo + hi) / 2


def raw_path(plumes: Path, row: dict) -> Path:
    return plumes / row["label_src"] / (Path(row["path"]).stem + ".tif")


def read(plumes: Path, row: dict) -> np.ndarray:
    with rasterio.open(raw_path(plumes, row)) as src:
        return src.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plumes", type=Path, required=True, help="raw images folder (positive/, negative/)")
    parser.add_argument("--index", type=Path, required=True, help="data_index/industrial_plumes.csv")
    parser.add_argument("--fasdd-index", type=Path, required=True)
    parser.add_argument("--fasdd-images", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--fit-samples", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    # FASDD-RS training images, loaded exactly like the training loader (PIL, convert RGB)
    with args.fasdd_index.open(encoding="utf-8") as stream:
        fasdd = [r for r in csv.DictReader(stream) if r["split"] == "train"]
    pick = rng.choice(len(fasdd), size=min(200, len(fasdd)), replace=False)
    fasdd_pixels = []
    for i in pick:
        name = Path(fasdd[i]["path"].replace("\\", "/")).name
        pixels = np.asarray(Image.open(args.fasdd_images / name).convert("RGB")).reshape(-1, 3)
        fasdd_pixels.append(pixels[rng.choice(len(pixels), 5000, replace=False)])
    fasdd_pixels = np.concatenate(fasdd_pixels)
    target = np.percentile(fasdd_pixels, PCT, axis=0)  # (len(PCT), 3)
    fasdd_median = np.percentile(fasdd_pixels, 50, axis=0)
    print(f"FASDD-RS train pixels: {len(fasdd_pixels)} from {len(pick)} images (of {len(fasdd)} train images)")
    print("FASDD-RS percentiles", PCT, "per channel R,G,B:")
    for p, row in zip(PCT, target):
        print(f"  p{p:<3d} {row.round(1).tolist()}")

    with args.index.open(encoding="utf-8") as stream:
        selected = list(csv.DictReader(stream))
    sample = [selected[i] for i in rng.choice(len(selected), args.samples, replace=False)]
    raw = [read(args.plumes, r) for r in sample]
    print(f"plume images sampled for scoring: {len(raw)} ({sum(r['label_src'] == 'positive' for r in sample)} positive)")

    results = []
    for gain, gamma in itertools.product(GAINS, GAMMAS):
        pixels = np.concatenate([convert(a, gain, gamma).reshape(-1, 3)[::7] for a in raw])
        results.append((distance(pixels, target), gain, gamma, float((pixels == 255).mean())))
    results.sort(key=lambda t: t[0])
    print("\nBASE candidates (gain, gamma): mean abs percentile difference to FASDD-RS (8-bit units) | share at 255")
    for score, gain, gamma, sat in results[:6]:
        print(f"  gain {gain:4.2f} gamma {gamma:4.2f}: {score:6.2f} | {sat:.4f}")
    base_score, gain, gamma, base_sat = results[0]
    print(f"\nBASE = gain {gain} gamma {gamma}, score {base_score:.2f}")

    # PER-CHANNEL version, fitted on plume TRAIN images only
    rng2 = np.random.default_rng(1)
    train_rows = [r for r in selected if r["split"] == "train"]
    fit_rows = [train_rows[i] for i in rng2.choice(len(train_rows), args.fit_samples, replace=False)]
    k, y0 = fit_per_channel([read(args.plumes, r) for r in fit_rows], gain, gamma, fasdd_median)
    pixels_pc = np.concatenate([convert_per_channel(a, gain, gamma, tuple(k), y0).reshape(-1, 3)[::7] for a in raw])
    pc_score, pc_sat = distance(pixels_pc, target), float((pixels_pc == 255).mean())
    pixels_base = np.concatenate([convert(a, gain, gamma).reshape(-1, 3)[::7] for a in raw])
    got_pc, got_base = np.percentile(pixels_pc, PCT, axis=0), np.percentile(pixels_base, PCT, axis=0)
    print(f"\nPER-CHANNEL: k (R,G,B) = {k.round(4).tolist()}, shoulder y0 = {y0:.4f}; fitted on {len(fit_rows)} plume train images")
    print(f"  score on the base's 400-image sample: {pc_score:.2f} (base {base_score:.2f}) | share at 255: {pc_sat:.4f} (base {base_sat:.4f})")
    adopt = pc_score <= base_score
    print(f"  adopt the per-channel version (score not worse than base): {adopt}")
    print("  percentiles per channel R,G,B:   per-channel | base | FASDD-RS")
    for p, a, b, t in zip(PCT, got_pc, got_base, target):
        print(f"  p{p:<3d} {a.round(0).astype(int).tolist()} | {b.round(0).astype(int).tolist()} | {t.round(0).astype(int).tolist()}")
    final = {"base": {"gain": gain, "gamma": gamma, "score": base_score},
             "per_channel": {"gain": gain, "gamma": gamma, "k": [float(x) for x in k], "y0": float(y0), "p99_target": P99_TARGET,
                             "score": pc_score, "adopted": bool(adopt)},
             "fasdd_rs_train_median": fasdd_median.tolist(), "fasdd_rs_train_percentiles": {str(p): t.tolist() for p, t in zip(PCT, target)}}
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    (args.preview_dir / "conversion_candidates.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    # preview sheets: 5 plume + 5 non-plume converted images beside 10 FASDD-RS images, as the model sees them
    pos = [r for r in selected if r["label_src"] == "positive" and r["split"] == "test"]
    neg = [r for r in selected if r["label_src"] == "negative" and r["split"] == "test"]
    chosen = [pos[i] for i in rng.choice(len(pos), 5, replace=False)] + [neg[i] for i in rng.choice(len(neg), 5, replace=False)]
    fas_pos = [r for r in fasdd if "smoke" in Path(r["path"].replace("\\", "/")).name.lower()]
    fas_neg = [r for r in fasdd if "neither" in Path(r["path"].replace("\\", "/")).name.lower()]
    fchosen = [fas_pos[i] for i in rng.choice(len(fas_pos), 5, replace=False)] + [fas_neg[i] for i in rng.choice(len(fas_neg), 5, replace=False)]
    chosen_raw = [read(args.plumes, r) for r in chosen]
    fasdd_tiles = [model_view(Image.open(args.fasdd_images / Path(r["path"].replace("\\", "/")).name).convert("RGB")) for r in fchosen]
    rows = [(f"plume BASE gain {gain} gamma {gamma}", [model_view(Image.fromarray(convert(a, gain, gamma))) for a in chosen_raw]),
            (f"plume PER-CHANNEL k={k.round(2).tolist()} y0={y0:.2f}",
             [model_view(Image.fromarray(convert_per_channel(a, gain, gamma, tuple(k), y0))) for a in chosen_raw]),
            ("FASDD-RS (left 5: smoke, right 5: neither)", fasdd_tiles)]
    canvas = Image.new("RGB", (10 * 192, len(rows) * 210), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, tiles) in enumerate(rows):
        draw.text((4, r * 210 + 2), label + ("   (left 5: plume, right 5: no plume)" if r < 2 else ""), fill="black")
        for j, tile in enumerate(tiles):
            canvas.paste(tile, (j * 192, r * 210 + 18))
    canvas.save(args.preview_dir / "preview_final_candidates.png")
    print(f"previews written to {args.preview_dir}")


if __name__ == "__main__":
    main()
