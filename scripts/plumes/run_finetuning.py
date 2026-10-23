"""Run the whole plume fine-tuning in the order and with the rules of external/PLUME_PROTOCOL.md (Section 7). Standard library only.

    python -u scripts/plumes/run_finetuning.py --dataset-root <dir with datasets/> --out-root <main checkout>/outputs/plume_finetune

Order: (1) RS teacher; its check (FASDD-RS val binary balanced accuracy drop > 2 points -> rerun with source-balanced batches; if
that also fails, STOP); (2) KD routed and KD uniform with the first teacher run that passes; (3) no-KD student, MobileNetV3-Large,
ResNet-50; (4) controls without plume images: KD routed (with the ORIGINAL CV / RS / UAV teachers) and MobileNetV3-Large.
Every model other than the teacher: if the binary recall on FASDD-RS val `both` images of the selected checkpoint drops by more
than 2 points from the before model, it is rerun with the class-balanced RS sampler into `<name>_balanced_rs` and both are kept.
The driver STOPS (writes STOP.txt and exits) when a run fails, when neither teacher run passes the balanced-accuracy check, or when
the `both`-recall rule triggers for the teacher: the protocol says to ask Amir in those cases. A run with a summary.json is skipped,
so the driver can be restarted. Seed 42; 4 data-loader workers (6 stalled on this Windows machine).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("D:/Thesis-Project/TUBITAK-1002")  # main checkout: outputs/ (checkpoints) live there, ignored by git
TEACHERS = {k: MAIN / f"outputs/teachers/FASDD_{k.upper()}/best.pt" for k in ("cv", "rs", "uav")}
START = {
    "kd_routed": MAIN / "outputs/retrain/kd_routed/seed42/best.pt",
    "kd_uniform": MAIN / "outputs/retrain/kd_uniform/seed42/best.pt",
    "nokd": MAIN / "outputs/retrain/nokd/seed42/best.pt",
    "mobilenet_v3_large": MAIN / "outputs/baselines/mobilenet_v3_large/seed42/best.pt",
    "resnet50": MAIN / "outputs/baselines/resnet50/seed42/best.pt",
    "teacher": TEACHERS["rs"],
}


class Stop(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path, decisions_path = args.out_root / "driver.log", args.out_root / "driver_decisions.json"
    decisions: dict = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def decide(key: str, value) -> None:
        decisions[key] = value
        decisions_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        log(f"DECISION {key}: {value}")

    def run(name: str, kind: str, extra: list[str]) -> dict:
        out = args.out_root / name / f"seed{args.seed}"
        summary_path = out / "summary.json"
        if summary_path.exists():
            log(f"skip {name} (summary.json exists)")
            return json.loads(summary_path.read_text())
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "scripts/finetune_rs_plumes.py"), "--kind", kind, "--checkpoint", str(START[kind]),
                   "--output-dir", str(out), "--dataset-root", str(args.dataset_root), "--seed", str(args.seed),
                   "--num-workers", str(args.num_workers)] + extra
        log(f"START {name}: {' '.join(command[3:])}")
        with (out / "run.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}).returncode
        if code != 0 or not summary_path.exists():
            raise Stop(f"{name} failed (exit code {code}); see {out / 'run.log'}")
        summary = json.loads(summary_path.read_text())
        log(f"DONE {name}: best epoch {summary['best_epoch']}, score {summary['best_selection_score']:.4f} "
            f"(before {summary['before_selection_score']:.4f}), FASDD-RS val BA drop {summary['fasdd_rs_val_ba_drop_points']:+.2f} pts, "
            f"both-recall drop {summary['both_recall_drop_points']:+.2f} pts, {summary['seconds'] / 60:.0f} min")
        return summary

    def with_fallback(name: str, kind: str, extra: list[str]) -> None:
        summary = run(name, kind, extra)
        if summary["fallback_triggered"]:
            decide(f"{name}_both_recall_fallback", f"both-recall dropped {summary['both_recall_drop_points']:.2f} pts (> 2): rerun with the "
                   "class-balanced RS sampler; both runs are kept")
            run(f"{name}_balanced_rs", kind, extra + ["--balanced-rs-sampler"])

    try:
        # 1. RS teacher and its check
        t1 = run("teacher_rs", "teacher", [])
        chosen, summary = "teacher_rs", t1
        if t1["ba_fallback_triggered"]:
            decide("teacher_ba_check", f"normal teacher run FAILED: FASDD-RS val BA drop {t1['fasdd_rs_val_ba_drop_points']:.2f} pts (> 2); "
                   "rerun with source-balanced batches")
            t2 = run("teacher_rs_srcbal", "teacher", ["--source-balanced-batches"])
            if t2["ba_fallback_triggered"]:
                raise Stop(f"neither teacher run passes the balanced-accuracy check (drops {t1['fasdd_rs_val_ba_drop_points']:.2f} and "
                           f"{t2['fasdd_rs_val_ba_drop_points']:.2f} pts): ask Amir before fine-tuning the KD students")
            chosen, summary = "teacher_rs_srcbal", t2
        else:
            decide("teacher_ba_check", f"normal teacher run passed: FASDD-RS val BA drop {t1['fasdd_rs_val_ba_drop_points']:.2f} pts (<= 2)")
        if summary["fallback_triggered"]:
            raise Stop(f"the both-recall rule triggered for the teacher run {chosen} (drop {summary['both_recall_drop_points']:.2f} pts): "
                       "the protocol says to ask Amir instead of starting a third teacher run")
        teacher_path = args.out_root / chosen / f"seed{args.seed}" / "best.pt"
        decide("teacher_used_by_kd_students", str(teacher_path))
        kd = ["--cv-teacher", str(TEACHERS["cv"]), "--rs-teacher", str(teacher_path), "--uav-teacher", str(TEACHERS["uav"])]

        # 2. KD students with the fine-tuned teacher
        with_fallback("kd_routed", "kd_routed", kd)
        with_fallback("kd_uniform", "kd_uniform", kd)
        # 3. students without a teacher, and the CNN baselines
        with_fallback("nokd", "nokd", [])
        with_fallback("mobilenet_v3_large", "mobilenet_v3_large", [])
        with_fallback("resnet50", "resnet50", [])
        # 4. controls: the same fine-tuning without plume images (KD control with the original teachers)
        original = ["--cv-teacher", str(TEACHERS["cv"]), "--rs-teacher", str(TEACHERS["rs"]), "--uav-teacher", str(TEACHERS["uav"])]
        run("kd_routed_noplume", "kd_routed", original + ["--no-plumes"])
        run("mobilenet_v3_large_noplume", "mobilenet_v3_large", ["--no-plumes"])
        log("ALL RUNS FINISHED")
    except Stop as stop:
        (args.out_root / "STOP.txt").write_text(f"{datetime.now().isoformat(timespec='seconds')}\n{stop}\n", encoding="utf-8")
        log(f"STOP: {stop}")
        sys.exit(2)


if __name__ == "__main__":
    main()
