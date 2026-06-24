from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from bootstrap import setup_path

ROOT = setup_path()

from firecls.deployment.artifacts import git_commit, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a target-specific TensorRT engine with trtexec.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/student_fp16.engine"))
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise SystemExit("trtexec was not found. Run this script in the pinned TensorRT container.")
    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)
    manifest_path = args.onnx.with_suffix(args.onnx.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = manifest["input"]["shape"]
    input_name = manifest["input"]["name"]
    channels, height, width = shape[1:]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def profile(batch: int) -> str:
        return f"{input_name}:{batch}x{channels}x{height}x{width}"

    command = [
        trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.output}",
        f"--minShapes={profile(args.min_batch)}",
        f"--optShapes={profile(args.opt_batch)}",
        f"--maxShapes={profile(args.max_batch)}",
        f"--memPoolSize=workspace:{args.workspace_mib}MiB",
        "--skipInference",
    ]
    if args.precision == "fp16":
        command.append("--fp16")
    subprocess.run(command, check=True)

    version = subprocess.run([trtexec, "--version"], check=True, text=True, capture_output=True)
    engine_manifest = {
        "artifact": str(args.output),
        "artifact_sha256": sha256_file(args.output),
        "source_onnx": str(args.onnx),
        "source_onnx_sha256": sha256_file(args.onnx),
        "source_checkpoint": manifest.get("source_checkpoint"),
        "source_checkpoint_sha256": manifest.get("source_checkpoint_sha256"),
        "source_git_commit": git_commit(ROOT),
        "classes": manifest["classes"],
        "input": manifest["input"],
        "output": manifest["output"],
        "precision": args.precision,
        "profile": {
            "min_batch": args.min_batch,
            "opt_batch": args.opt_batch,
            "max_batch": args.max_batch,
        },
        "trtexec_version": (version.stdout + version.stderr).strip(),
        "portability_warning": "Rebuild on each target GPU/platform with this pinned container.",
    }
    output_manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_json(output_manifest, engine_manifest)
    print(f"Built TensorRT engine: {args.output}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
