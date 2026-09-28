"""CPU-only checks for the CNN classifier baselines; no dataset or weight download needed."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from firecls.baselines.export import export_scores_model
from firecls.baselines.families.cnn import build_cnn, preprocessing
from firecls.baselines.protocol import CLASSES
from firecls.baselines.registry import load_baseline

REPO = Path(__file__).resolve().parents[1]
COLORS = {"fire": (230, 60, 20), "smoke": (150, 150, 150), "both": (200, 120, 90), "neither": (20, 120, 40)}


def make_fixture(root: Path, per_class: int = 2) -> None:
    """A miniature FASDD: three domains, every class in every split, index CSVs in data_index/."""
    from firecls.config import get_dataset_specs

    (root / "data_index").mkdir(parents=True)
    for domain, spec in get_dataset_specs(root).items():
        spec.images_root.mkdir(parents=True)
        with (root / "data_index" / f"{spec.name.lower()}_tdml.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
            writer.writeheader()
            for split in ("train", "val", "test"):
                for label in CLASSES:
                    for i in range(per_class):
                        path = spec.images_root / f"{label}_{split}_{i}.jpg"
                        Image.new("RGB", (96, 64), COLORS[label]).save(path)
                        # Windows-style path, as in the committed indices, to exercise re-basing.
                        writer.writerow({"path": f"E:\\x\\images\\{path.name}", "label": label, "split": split})


class CnnModelTests(unittest.TestCase):
    def test_heads_have_four_outputs(self):
        for arch in ("resnet50", "mobilenet_v3_large"):
            model = build_cnn(arch, pretrained=False).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(2, 3, 192, 192)).shape), (2, 4))

    def test_export_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.pt"
            model = build_cnn("mobilenet_v3_large", pretrained=False)
            torch.save({"model": model.state_dict(), "arch": "mobilenet_v3_large", "classes": CLASSES}, ckpt)
            loaded = load_baseline("cnn", ckpt)
            onnx_path = export_scores_model(
                loaded.model, Path(d) / "m.onnx", loaded.preprocessing, ckpt, "cnn", "mobilenet", "official"
            )
            manifest = json.loads(onnx_path.with_suffix(".onnx.manifest.json").read_text())
            self.assertEqual(manifest["input"]["shape"], ["batch", 3, 192, 192])
            self.assertEqual(manifest["preprocessing"], preprocessing())
            self.assertTrue(manifest["verification"]["top1_match"])


class CnnPipelineTest(unittest.TestCase):
    """train -> export -> evaluate -> benchmark -> table, exactly as run_all.sh chains them."""

    def run_script(self, *args):
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
        result = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout[-2000:] + result.stderr[-4000:])
        return result.stdout

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_fixture(root)
            out = root / "run"
            self.run_script(
                "scripts/cnn/train_classifier.py", "--arch", "mobilenet_v3_large", "--seed", "42", "--epochs", "2",
                "--batch-size", "4", "--num-workers", "0", "--no-pretrained",
                "--dataset-root", root, "--index-root", root, "--output-dir", out,
            )
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(len(history["records"]), 2)
            self.assertEqual(set(history["records"][0]["val_acc"]), {"cv", "rs", "uav"})

            # --resume with the same budget is a no-op rather than extra training.
            self.run_script(
                "scripts/cnn/train_classifier.py", "--arch", "mobilenet_v3_large", "--seed", "42", "--epochs", "2",
                "--batch-size", "4", "--num-workers", "0", "--no-pretrained", "--resume",
                "--dataset-root", root, "--index-root", root, "--output-dir", out,
            )
            self.assertEqual(len(json.loads((out / "history.json").read_text())["records"]), 2)

            ckpt, onnx = out / "best.pt", root / "m.onnx"
            self.run_script("scripts/export_baseline.py", "--family", "cnn", "--checkpoint", ckpt, "--output", onnx)
            eval_json, bench_json = root / "mbv3_seed42_eval.json", root / "mbv3_seed42_bench.json"
            # evaluate_baseline reads data_index from the repository, so point it at the fixture copy.
            self.run_script(
                "scripts/evaluate_baseline.py", "--family", "cnn", "--name", "mbv3_seed42", "--checkpoint", ckpt,
                "--onnx", onnx, "--dataset-root", root, "--num-workers", "0", "--output", eval_json,
                "--index-root", root,
            )
            result = json.loads(eval_json.read_text())
            self.assertEqual(set(result["models"]), {"mbv3_seed42_pt", "mbv3_seed42_onnx"})
            for domain in ("cv", "rs", "uav"):
                self.assertEqual(result["models"]["mbv3_seed42_pt"]["domains"][domain]["metrics"]["samples"], 8)
                self.assertEqual(result["runtime_agreement"]["onnx"][domain]["top1_agreement"], 1.0)

            self.run_script(
                "scripts/benchmark_baseline.py", "--family", "cnn", "--name", "mbv3_seed42", "--checkpoint", ckpt,
                "--onnx", onnx, "--batch-sizes", "1", "--warmup", "1", "--runs", "3", "--output", bench_json,
            )
            table = self.run_script(
                "scripts/collect_results_table.py", "--eval", eval_json, "--bench", bench_json,
                "--accuracy-runtime", "onnx", "--latency-runtime", "onnx", "--output", root / "table",
            )
            self.assertIn(r"mbv3 &", table)
            self.assertTrue((root / "table.csv").exists())


def hub_reachable() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
        return True
    except Exception:
        return False


@unittest.skipUnless(hub_reachable(), "Hugging Face Hub unreachable; SwinV2 no-KD path needs the model config")
class SwinNoKdTest(unittest.TestCase):
    def test_checkpoint_loads_with_student_family(self):
        sys.path.insert(0, str(REPO / "scripts"))
        sys.path.insert(0, str(REPO / "scripts" / "cnn"))
        import train_classifier

        model, mean, std = train_classifier.build_model("swinv2_base_nokd", pretrained=False)
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.pt"
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "model_name": train_classifier.STUDENT_BACKBONE}, ckpt)
            loaded = load_baseline("swinv2", ckpt)
            with torch.no_grad():
                self.assertEqual(tuple(loaded.model(torch.randn(1, 3, 192, 192)).shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
