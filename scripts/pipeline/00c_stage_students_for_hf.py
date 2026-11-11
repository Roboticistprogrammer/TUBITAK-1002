"""Strip optimizer state, copy history, export ONNX for the 7 new students into the HF models clone."""
import json, shutil, subprocess, sys, hashlib
from pathlib import Path
import torch

MAIN = Path("D:/Thesis-Project/TUBITAK-1002")
HF = Path("D:/Thesis-Project/models")
WT = MAIN / ".claude/worktrees/students-seeds"
PY = MAIN / ".venv/Scripts/python.exe"
JOBS = [("kd_routed", "kd_routed", 43), ("kd_routed", "kd_routed", 44),
        ("kd_uniform", "kd_uniform", 43), ("kd_uniform", "kd_uniform", 44),
        ("nokd_noamp", "nokd_noamp", 42), ("nokd_noamp", "nokd_noamp", 43), ("nokd_noamp", "nokd_noamp", 44)]
for src_name, hf_name, seed in JOBS:
    src = MAIN / f"outputs/retrain/{src_name}/seed{seed}"
    dst = HF / f"students/seed{seed}/{hf_name}"
    dst.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
    dropped = [k for k in ("optimizer", "scheduler") if ckpt.pop(k, None) is not None]
    torch.save(ckpt, dst / "best.pt")
    hist = src / "history.json"
    if not hist.exists():  # seed-43 KD histories are tracked only in the worktree
        hist = WT / f"outputs/retrain/{src_name}/seed{seed}/history.json"
    shutil.copy(hist, dst / "history.json")
    onnx = dst / f"{hf_name}_seed{seed}.onnx"
    env = {**__import__("os").environ, "PYTHONPATH": str(WT / "src")}
    subprocess.run([str(PY), str(WT / "scripts/export_student.py"), "--checkpoint", str(dst / "best.pt"), "--output", str(onnx)],
                   cwd=WT, env={**env}, check=True)
    m = json.loads(Path(str(onnx) + ".manifest.json").read_text(encoding="utf-8"))
    print("EXPORTED", hf_name, seed, "epoch", ckpt["epoch"], "avg_val", round(ckpt["avg_val"], 4), "dropped", dropped,
          m["verification"], flush=True)
