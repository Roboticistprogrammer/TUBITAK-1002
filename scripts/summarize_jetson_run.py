"""Convert a ``jetson_trtexec_benchmark.sh`` output folder into the benchmark JSON schema.

trtexec's ``--exportTimes`` file lists per-inference timings; we report end-to-end latency
(host-to-device + compute + device-to-host) so the numbers are comparable with
``scripts/benchmark_baseline.py``. Power comes from the tegrastats log recorded alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap import setup_path

setup_path()

from firecls.baselines.power import parse_power_mw
from firecls.deployment.artifacts import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", default="unknown")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def end_to_end_ms(record: dict) -> float:
    if "latencyMs" in record:
        return float(record["latencyMs"])
    parts = ("h2dMs", "computeMs", "d2hMs")
    return float(sum(record.get(key, 0.0) for key in parts))


def main() -> None:
    args = parse_args()
    times = json.loads((args.run_dir / "times.json").read_text(encoding="utf-8"))
    values = np.asarray([end_to_end_ms(r) for r in times if isinstance(r, dict)])
    compute = np.asarray([float(r.get("computeMs", np.nan)) for r in times if isinstance(r, dict)])
    readings = [p for p in map(parse_power_mw, (args.run_dir / "tegrastats.log").read_text().splitlines()) if p]
    power = [value for _, value in readings]
    rail = readings[0][0] if readings else None

    mean_ms = float(values.mean())
    entry = {
        "batch_size": args.batch,
        "runs": int(values.size),
        "mean_batch_ms": mean_ms,
        "p50_batch_ms": float(np.percentile(values, 50)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "p99_batch_ms": float(np.percentile(values, 99)),
        "std_batch_ms": float(values.std()),
        "mean_compute_ms": float(np.nanmean(compute)),
        "mean_ms_per_image": mean_ms / args.batch,
        "throughput_images_per_second": 1000.0 * args.batch / mean_ms,
    }
    if power:
        mean_mw = float(np.mean(power))
        entry["power"] = {"rail": rail, "samples": len(power), "mean_mw": mean_mw, "max_mw": float(np.max(power))}
        entry["energy_mj_per_image"] = mean_mw * (mean_ms / args.batch) / 1000.0

    payload = {
        "baseline": {"name": args.name, "family": args.family},
        "protocol": {"tool": "trtexec", "timing_scope": "end-to-end per inference (H2D + compute + D2H)"},
        "runtimes": {"engine": {"artifact": str(args.run_dir), "batches": {str(args.batch): entry}}},
    }
    output = args.output or args.run_dir / "benchmark.json"
    write_json(output, payload)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
