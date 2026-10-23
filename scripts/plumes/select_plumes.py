"""Select 3,000 plume + 3,000 non-plume images and split them by site group (external/PLUME_PROTOCOL.md, Section 3.2).

    python -I scripts/plumes/select_plumes.py --table datasets/IndustrialSmokePlumes/raw/image_table.csv ^
        --out data_index/industrial_plumes.csv

Input is the table written by `site_table.py`. Rules, all fixed before looking at any model result:

* Images 120 x 119 px (1 plume, 23 non-plume) are excluded; every other image is 120 x 120.
* Site = the leading number of the file name. Site ids whose image centres are closer than 2.5 km (image diagonal 1.7 km
  plus centre jitter) are merged into one **site group**, because their footprints can overlap; some ids are the same
  place. Whole groups go to one split, so no place appears in two splits. Seed 42.
* Plume images: all groups that have both plume and non-plume images are used whole (so non-plume images exist from the same
  place); positive-only groups are then added whole in seeded random order until 3,000, and the last group is trimmed at
  random. Groups are then assigned to train / val / test (about 70 / 15 / 15 of the plume images) by a seeded
  largest-first greedy fill.
* Non-plume images: as many as plume images in each split (balanced inside every split). Every plume image is paired with
  one distinct non-plume image of the same split by an optimal assignment (cost = 2 x calendar-month gap + 1 if the
  non-plume image is from another group). Month gets the larger weight because plumes are much more common in winter
  (41-42% of images in December and January, 9-10% in July and August), so month would otherwise predict the label.
  The pool is all non-plume images of the groups in that split, including groups without plume images (spread over the
  splits at random).

Output columns: path,site,group,month,label_src,label,split. `path` is the converted 8-bit RGB PNG,
datasets/IndustrialSmokePlumes/rgb/<label_src>/<name>.png (written by convert_to_rgb.py); `group` is the 2.5 km site group,
`month` the acquisition month (1-12), `label_src` the Zenodo folder (positive / negative) and `label` the 4-class training
label (smoke / neither). The CSV is the list of selected files. A per-split report goes to <out>.meta.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SEED = 42
TARGET_PLUME = 3000
SPLITS = ("train", "val", "test")
SPLIT_SHARE = (0.70, 0.15, 0.15)
GROUP_KM = 2.5
RGB_DIR = "datasets/IndustrialSmokePlumes/rgb"


def month_gap(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 12 - d)


def site_groups(rows: list[dict]) -> dict[str, int]:
    sites = sorted({r["site"] for r in rows}, key=int)
    lon, lat = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        lon[r["site"]].append(float(r["lon"]))
        lat[r["site"]].append(float(r["lat"]))
    xy = np.array([[np.median(lon[s]) * 111.32 * np.cos(np.radians(np.median(lat[s]))), np.median(lat[s]) * 110.57]
                   for s in sites])
    pairs = np.array(sorted(cKDTree(xy).query_pairs(GROUP_KM)), dtype=int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(sites),) * 2)
    _, labels = connected_components(graph, directed=False)
    return {s: int(labels[i]) for i, s in enumerate(sites)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--month-weight", type=float, default=2.0, help="cost per month of calendar gap")
    parser.add_argument("--group-weight", type=float, default=1.0, help="cost of taking a non-plume image from another group")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    all_rows = list(csv.DictReader(args.table.open(encoding="utf-8")))
    rows = [r for r in all_rows if (r["height"], r["width"]) == ("120", "120")]
    excluded = len(all_rows) - len(rows)
    group_of = site_groups(rows)
    for r in rows:
        r["group"] = group_of[r["site"]]
        r["month"] = int(r["date"][5:7])

    plume, plain = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        (plume if r["label_src"] == "positive" else plain)[r["group"]].append(r)

    # 1. which plume images: all mixed groups whole, then positive-only groups in random order, last one trimmed
    mixed = sorted(g for g in plume if g in plain)
    only = sorted(g for g in plume if g not in plain)
    chosen_groups = list(mixed)
    total = sum(len(plume[g]) for g in mixed)
    for g in rng.permutation(only):
        if total >= TARGET_PLUME:
            break
        chosen_groups.append(int(g))
        total += len(plume[int(g)])
    selected = {g: list(plume[g]) for g in chosen_groups}
    excess = total - TARGET_PLUME
    if excess > 0:  # trim the last group (random images) so the total is exactly 3,000
        last = chosen_groups[-1]
        keep = sorted(rng.choice(len(selected[last]), len(selected[last]) - excess, replace=False))
        selected[last] = [selected[last][i] for i in keep]
    assert sum(len(v) for v in selected.values()) == TARGET_PLUME

    # 2. groups -> splits: plume groups largest-first into the split furthest below its target, then the rest at random
    targets = [s * TARGET_PLUME for s in SPLIT_SHARE]
    filled = [0, 0, 0]
    split_of: dict[int, str] = {}
    order = sorted(selected, key=lambda g: (-len(selected[g]), rng.random()))
    for g in order:
        k = int(np.argmax([targets[i] - filled[i] for i in range(3)]))
        split_of[g] = SPLITS[k]
        filled[k] += len(selected[g])
    others = [g for g in sorted(set(plain) | set(plume)) if g not in split_of]
    for g in rng.permutation(others):
        split_of[int(g)] = SPLITS[int(rng.choice(3, p=SPLIT_SHARE))]

    # 3. non-plume images, balanced per split: optimal one-to-one assignment of a non-plume image to every plume image of
    #    the split (scipy.optimize.linear_sum_assignment), cost = 2 x (calendar-month gap, 0..6) + 1 if from another group.
    #    A month gap costs twice as much as leaving the group, because month predicts the label far more strongly.
    chosen_plain, same_group, month_gaps = [], 0, []
    for split in SPLITS:
        plumes = [p for g in selected if split_of[g] == split for p in selected[g]]
        pool = [n for g in split_of if split_of[g] == split for n in plain.get(g, [])]
        p_month = np.array([p["month"] for p in plumes])[:, None]
        n_month = np.array([n["month"] for n in pool])[None, :]
        gap = np.abs(p_month - n_month)
        gap = np.minimum(gap, 12 - gap)
        diff_group = (np.array([p["group"] for p in plumes])[:, None] != np.array([n["group"] for n in pool])[None, :])
        cost = args.month_weight * gap + args.group_weight * diff_group + rng.random(gap.shape) * 1e-3  # noise: tie-break
        rows_i, cols_j = linear_sum_assignment(cost)
        assert len(rows_i) == len(plumes) and len(set(cols_j)) == len(plumes)
        for i, j in zip(rows_i, cols_j):
            chosen_plain.append(pool[j])
            same_group += int(not diff_group[i, j])
            month_gaps.append(int(gap[i, j]))
    same_month = sum(1 for g in month_gaps if g == 0)

    # 4. write the index
    out_rows = []
    for g, items in selected.items():
        out_rows += [(r, "smoke", split_of[g]) for r in items]
    out_rows += [(r, "neither", split_of[r["group"]]) for r in chosen_plain]
    out_rows.sort(key=lambda t: (t[2], t[0]["label_src"], t[0]["file"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["path", "site", "group", "month", "label_src", "label", "split"])
        for r, label, split in out_rows:
            stem = r["file"][: -len(".tif")]
            writer.writerow([f"{RGB_DIR}/{r['label_src']}/{stem}.png", r["site"], r["group"], r["month"],
                             r["label_src"], label, split])

    # report
    counts = collections.Counter((split, label) for _, label, split in out_rows)
    leak = collections.defaultdict(set)
    for r, _, split in out_rows:
        leak[r["group"]].add(split)
    month = {lab: collections.Counter() for lab in ("smoke", "neither")}
    for r, label, _ in out_rows:
        month[label][r["month"]] += 1
    report = {
        "seed": SEED, "excluded_120x119": excluded, "groups_total": len(split_of), "groups_with_selected_images": len(leak),
        "groups_in_more_than_one_split": sum(1 for v in leak.values() if len(v) > 1),
        "images": {f"{s}/{l}": counts[(s, l)] for s in SPLITS for l in ("smoke", "neither")},
        "plume_images_in_mixed_groups": sum(len(selected[g]) for g in mixed if g in selected),
        "non_plume_from_same_group": same_group, "non_plume_from_other_group": TARGET_PLUME - same_group,
        "non_plume_same_month_as_matched_plume": same_month, "mean_month_gap": float(np.mean(month_gaps)),
        "month_histogram": {lab: [month[lab][m] for m in range(1, 13)] for lab in month},
        "sites_selected": len({r["site"] for r, _, _ in out_rows}),
    }
    plume_only = set(only)  # groups that have plume images but no non-plume image anywhere in the data
    per_split = {}
    for s in SPLITS:
        members = [(r, label) for r, label, split in out_rows if split == s]
        groups = {r["group"] for r, _ in members}
        month_s = {lab: [0] * 12 for lab in ("smoke", "neither")}
        for r, label in members:
            month_s[label][r["month"] - 1] += 1
        per_split[s] = {
            "groups": len(groups), "site_ids": len({r["site"] for r, _ in members}),
            "plume_images": counts[(s, "smoke")], "non_plume_images": counts[(s, "neither")],
            "plume_only_groups": len(groups & plume_only),
            "plume_images_from_plume_only_groups": sum(1 for r, label in members if label == "smoke" and r["group"] in plume_only),
            "month_histogram": month_s,
        }
    report["per_split"] = per_split
    print(json.dumps(report, indent=1))
    (args.out.with_suffix(".meta.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
