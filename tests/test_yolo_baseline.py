"""YOLO baseline (Vazquez et al. adaptation): wrapper, export, loader and dataset preparation.

Runs on CPU without FASDD and without network: detectors are random-init ``yolov8n.yaml``
models and the dataset is a synthetic FASDD-shaped tree written to a temporary directory.
"""
import copy
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]

try:
    import ultralytics  # noqa: F401

    HAVE_ULTRALYTICS = True
except ImportError:  # the baseline is optional; the rest of the suite must still run
    HAVE_ULTRALYTICS = False

from firecls.baselines.protocol import CLASSES

DOMAIN_DIRS = {"cv": ("FASDD_CV", "FASDD_CV", "images"), "rs": ("FASDD_RS", "images"), "uav": ("FASDD_UAV", "images")}
DOMAIN_LABELS = {"cv": ["fire", "smoke", "both", "neither"], "rs": ["smoke", "both", "neither"], "uav": ["fire", "smoke", "both", "neither"]}
SUFFIX = {"cv": ".jpg", "rs": ".tif", "uav": ".jpg"}


def load_script(relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_synthetic_fasdd(root: Path, per_split: int = 2, size: tuple[int, int] = (96, 64)) -> dict[str, Path]:
    """FASDD-shaped tree: ``root/datasets/...`` images, ``root/data_index`` CSVs with Windows paths,
    COCO json boxes for CV/UAV and a YOLO label dir (no classes.txt) for RS. Returns box sources."""
    width, height = size
    (root / "data_index").mkdir(parents=True, exist_ok=True)
    sources = {}
    for domain, parts in DOMAIN_DIRS.items():
        images_root = root / "datasets" / Path(*parts)
        images_root.mkdir(parents=True, exist_ok=True)
        rows, coco_images, coco_annotations = ["path,label,split"], [], []
        yolo_dir = root / "boxes" / f"{domain}_yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        for split in ("train", "val", "test"):
            for _ in range(per_split):
                for label in DOMAIN_LABELS[domain]:
                    stem = f"{label}_{domain.upper()}{index:06d}"
                    index += 1
                    image = Image.new("RGB", (width, height), (40, 90, 40))
                    draw = ImageDraw.Draw(image)
                    boxes = []
                    if label in ("fire", "both"):
                        boxes.append(("fire", 5, 5, 30, 25))
                        draw.rectangle((5, 5, 35, 30), fill=(250, 60, 0))
                    if label in ("smoke", "both"):
                        boxes.append(("smoke", 40, 20, 40, 30))
                        draw.rectangle((40, 20, 80, 50), fill=(180, 180, 180))
                    image.save(images_root / f"{stem}{SUFFIX[domain]}")
                    windows = "E:\\thesis\\" + "\\".join(("datasets", *parts)) + f"\\{stem}{SUFFIX[domain]}"
                    rows.append(f"{windows},{label},{split}")
                    coco_images.append({"id": index, "file_name": f"{stem}{SUFFIX[domain]}", "width": width, "height": height})
                    lines = []
                    for cls, x, y, w, h in boxes:
                        coco_annotations.append(
                            {"id": len(coco_annotations) + 1, "image_id": index, "category_id": 1 if cls == "fire" else 2, "bbox": [x, y, w, h]}
                        )
                        cx, cy = (x + w / 2) / width, (y + h / 2) / height
                        lines.append(f"{0 if cls == 'fire' else 1} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
                    if lines or index % 2:  # some "neither" images have no label file at all
                        (yolo_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        name = {"cv": "fasdd_cv", "rs": "fasdd_rs", "uav": "fasdd_uav"}[domain]
        (root / "data_index" / f"{name}_tdml.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        if domain == "rs":
            sources[domain] = yolo_dir
        else:
            coco = root / "boxes" / f"{domain}_coco.json"
            coco.write_text(json.dumps({
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}],
            }), encoding="utf-8")
            sources[domain] = coco
    return sources


def random_detector(seed: int = 0):
    """Random-init YOLOv8n (nc=2) whose outputs depend on the image.

    With PyTorch's default init the activations of a fresh YOLO shrink to ~1e-7 before the head,
    so the class scores would reflect only the per-level biases. Re-estimating the BatchNorm
    statistics on one random batch restores unit-scale features and image-dependent scores.
    """
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(seed)
    detector = DetectionModel("yolov8n.yaml", nc=2, verbose=False)
    detector.names = {0: "fire", 1: "smoke"}
    for module in detector.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = None  # cumulative average: one batch sets the statistics
            module.reset_running_stats()
    detector.train()
    with torch.no_grad():
        detector(torch.rand(4, 3, 256, 256, generator=torch.Generator().manual_seed(seed + 100)))
    return detector.eval()


def thresholded(p_fire: float, p_smoke: float, t_fire: float, t_smoke: float) -> str:
    return {(True, False): "fire", (False, True): "smoke", (True, True): "both", (False, False): "neither"}[
        (bool(p_fire >= t_fire), bool(p_smoke >= t_smoke))
    ]


def save_like_ultralytics(detector, path: Path, model_arg: str = "yolov8n.pt", seed: int = 43, epoch: int = 3) -> None:
    """Write a checkpoint the way ``BaseTrainer.save_model`` does, then finalise it with
    ``strip_optimizer`` exactly as ``BaseTrainer.final_eval`` does for ``best.pt``."""
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT
    from ultralytics.utils.torch_utils import strip_optimizer

    torch.save(
        {
            "epoch": epoch - 1,
            "best_fitness": 0.1,
            "model": None,
            "ema": copy.deepcopy(detector).half(),
            "updates": 10,
            "optimizer": {"state": {}, "param_groups": []},
            "scaler": {},
            "train_args": {**DEFAULT_CFG_DICT, "model": model_arg, "seed": seed, "data": "fasdd_all.yaml"},
            "train_metrics": {"fitness": 0.1},
            "train_results": {"epoch": list(range(1, epoch + 1)), "metrics/mAP50-95(B)": [0.0] * epoch},
            "version": ultralytics.__version__,
        },
        path,
    )
    strip_optimizer(path)


@unittest.skipUnless(HAVE_ULTRALYTICS, "ultralytics not installed")
class YoloWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from firecls.baselines.families.yolo import YoloPresenceScores

        cls.images = torch.rand(8, 3, 320, 320, generator=torch.Generator().manual_seed(1))
        reference = random_detector()
        with torch.inference_mode():
            decoded = reference(cls.images)[0]  # standard Ultralytics eval output: (decoded, raw) tuple
        cls.reference_presence = decoded[:, 4].amax(1), decoded[:, 5].amax(1)
        probe = YoloPresenceScores(random_detector()).eval()
        with torch.inference_mode():
            p_fire, p_smoke = probe.presence(cls.images)
        # Median thresholds give a mixed 4-way decision and put one image exactly on each threshold.
        cls.t_fire, cls.t_smoke = float(p_fire.median()), float(p_smoke.median())
        cls.wrapper = YoloPresenceScores(random_detector(), cls.t_fire, cls.t_smoke).eval()

    def test_presence_matches_unfused_eval_output(self):
        with torch.inference_mode():
            p_fire, p_smoke = self.wrapper.presence(self.images)
        torch.testing.assert_close(p_fire, self.reference_presence[0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(p_smoke, self.reference_presence[1], atol=1e-5, rtol=1e-4)

    def test_scores_shape_and_argmax_is_thresholded_decision(self):
        with torch.inference_mode():
            scores = self.wrapper(self.images)
            p_fire, p_smoke = self.wrapper.presence(self.images)
        self.assertEqual(tuple(scores.shape), (8, len(CLASSES)))
        predicted = [CLASSES[i] for i in scores.argmax(1).tolist()]
        expected = [thresholded(f, s, self.t_fire, self.t_smoke) for f, s in zip(p_fire.tolist(), p_smoke.tolist())]
        self.assertEqual(predicted, expected)
        self.assertGreater(len(set(predicted)), 1)

    def test_class_indices(self):
        from firecls.baselines.families.yolo import class_indices

        self.assertEqual(class_indices({0: "fire", 1: "smoke"}), (0, 1))
        self.assertEqual(class_indices({0: "Smoke", 1: "Fire"}), (1, 0))
        with self.assertRaises(ValueError):
            class_indices({0: "0", 1: "1"})
        with self.assertRaises(ValueError):
            class_indices({0: "fire", 1: "smoke", 2: "person"})

    def test_onnx_export_verifies_with_dynamic_batch(self):
        import onnxruntime as ort

        from firecls.baselines.export import export_scores_model
        from firecls.baselines.preprocessing import letterbox_spec

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            export_model = getattr(self.wrapper, "export_module", lambda: self.wrapper)()
            onnx_path = export_scores_model(
                export_model, Path(tmp) / "yolo.onnx", letterbox_spec(640), checkpoint,
                family="yolo", model_name="yolov8n_scratch", source="reimplemented",
                extra={"thresholds": {"t_fire": self.t_fire, "t_smoke": self.t_smoke}},
            )
            manifest = json.loads(onnx_path.with_suffix(".onnx.manifest.json").read_text())
            self.assertTrue(manifest["verification"]["top1_match"])
            self.assertEqual(manifest["input"]["shape"], ["batch", 3, 640, 640])
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            single = torch.rand(1, 3, 640, 640)
            onnx_scores = session.run(["logits"], {"pixel_values": single.numpy()})[0]
            with torch.inference_mode():
                np.testing.assert_allclose(onnx_scores, self.wrapper(single).numpy(), atol=1e-4)


@unittest.skipUnless(HAVE_ULTRALYTICS, "ultralytics not installed")
class YoloLoaderTests(unittest.TestCase):
    def test_round_trip_ultralytics_checkpoint(self):
        from firecls.baselines.families.yolo import THRESHOLDS_FILE, YoloPresenceScores
        from firecls.baselines.registry import available_families, load_baseline
        from firecls.deployment.artifacts import sha256_file

        self.assertIn("yolo", available_families())
        detector = random_detector(3).half().float()  # exactly representable in the FP16 checkpoint
        images = torch.rand(2, 3, 256, 256)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "weights" / "best.pt"
            checkpoint.parent.mkdir()
            save_like_ultralytics(copy.deepcopy(detector), checkpoint)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_baseline("yolo", checkpoint, "cpu")
            self.assertTrue(any(THRESHOLDS_FILE in str(w.message) for w in caught))
            meta = loaded.metadata
            self.assertEqual(meta["classes"], CLASSES)
            self.assertEqual(meta["model_name"], "yolov8n_coco")
            self.assertEqual(meta["source"], "reimplemented")
            self.assertEqual(meta["thresholds"], {"t_fire": 0.5, "t_smoke": 0.5})
            self.assertFalse(meta["thresholds_calibrated"])
            self.assertEqual((meta["train_seed"], meta["epoch"]), (43, 3))
            self.assertEqual(loaded.preprocessing["mode"], "letterbox")
            self.assertEqual(loaded.preprocessing["size"], 640)
            with torch.inference_mode():
                expected = YoloPresenceScores(detector).eval()(images)
                np.testing.assert_allclose(loaded.model(images).numpy(), expected.numpy(), atol=1e-5)

            thresholds = checkpoint.parent / THRESHOLDS_FILE
            thresholds.write_text(json.dumps({"t_fire": 0.3, "t_smoke": 0.7, "checkpoint_sha256": sha256_file(checkpoint)}))
            loaded = load_baseline("yolo", checkpoint, "cpu")
            self.assertEqual(loaded.metadata["thresholds"], {"t_fire": 0.3, "t_smoke": 0.7})
            self.assertTrue(loaded.metadata["thresholds_calibrated"])
            self.assertEqual((loaded.model.head.t_fire, loaded.model.head.t_smoke), (0.3, 0.7))

            thresholds.write_text(json.dumps({"t_fire": 0.3, "t_smoke": 0.7, "checkpoint_sha256": "0" * 64}))
            with self.assertRaises(ValueError):
                load_baseline("yolo", checkpoint, "cpu")

    def test_scratch_run_name(self):
        from firecls.baselines.families.yolo import describe_run

        self.assertEqual(describe_run({"model": "yolov5nu.yaml"})["model_name"], "yolov5n_scratch")
        self.assertEqual(describe_run({"model": "/w/yolo11n.pt"})["model_name"], "yolo11n_coco")


@unittest.skipUnless(HAVE_ULTRALYTICS, "ultralytics not installed")
class PrepareFasddYoloTests(unittest.TestCase):
    def run_prepare(self, root: Path, sources: dict, *extra: str) -> dict:
        prepare = load_script("scripts/yolo/prepare_fasdd_yolo.py")
        return prepare.main([
            "--dataset-root", str(root), "--index-root", str(root), "--output", str(root / "datasets_yolo"),
            "--cv-boxes", str(sources["cv"]), "--rs-boxes", str(sources["rs"]), "--uav-boxes", str(sources["uav"]),
            "--yolo-class-names", "fire", "smoke", *extra,
        ])

    def test_builds_consistent_ultralytics_dataset(self):
        import yaml
        from ultralytics.data.dataset import YOLODataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = make_synthetic_fasdd(root)
            summary = self.run_prepare(root, sources)
            out = root / "datasets_yolo"
            for domain, report in summary["domains"].items():
                self.assertEqual(report["mismatches"], 0, domain)
            self.assertGreater(summary["domains"]["rs"]["images_without_annotation_entry"], 0)

            combined = yaml.safe_load((out / "fasdd_all.yaml").read_text())
            self.assertEqual(combined["train"], ["cv_train.txt", "rs_train.txt", "uav_train.txt"])
            self.assertEqual(combined["val"], ["cv_val.txt", "rs_val.txt", "uav_val.txt"])
            self.assertEqual(combined["names"], {0: "fire", 1: "smoke"})
            self.assertEqual(yaml.safe_load((out / "fasdd_rs.yaml").read_text())["test"], "rs_test.txt")

            entries = (out / "cv_train.txt").read_text().split()
            self.assertEqual(len(entries), 8)
            first = out / entries[0][2:]
            self.assertTrue(first.name.startswith("cv_"))
            self.assertTrue(first.resolve().is_relative_to((root / "datasets").resolve()))

            fire_label = out / "labels" / "cv" / "train" / "cv_fire_CV000000.txt"
            cls, cx, cy, w, h = map(float, fire_label.read_text().split())
            self.assertEqual(cls, 0)
            np.testing.assert_allclose([cx, cy, w, h], [20 / 96, 17.5 / 64, 30 / 96, 25 / 64], atol=1e-5)
            self.assertEqual((out / "labels" / "uav" / "val" / "uav_neither_UAV000011.txt").read_text(), "")

            # Ultralytics must find the labels through its /images/ -> /labels/ rule.
            data = {"names": {0: "fire", 1: "smoke"}, "nc": 2, "channels": 3}
            dataset = YOLODataset(img_path=[str(out / f"{d}_val.txt") for d in ("cv", "rs", "uav")], data=data,
                                  imgsz=64, augment=False, prefix="test: ")
            self.assertEqual(len(dataset), 8 + 6 + 8)
            box_counts = {Path(l["im_file"]).stem: len(l["cls"]) for l in dataset.labels}
            self.assertEqual(box_counts["rs_both_RS000010"], 2)
            self.assertEqual(box_counts["uav_neither_UAV000011"], 0)

    def test_fails_on_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = make_synthetic_fasdd(root)
            payload = json.loads(sources["cv"].read_text())
            payload["annotations"] = [a for a in payload["annotations"] if a["category_id"] != 2]  # drop smoke
            sources["cv"].write_text(json.dumps(payload))
            with self.assertRaises(SystemExit):
                self.run_prepare(root, sources)
            report = json.loads((root / "datasets_yolo" / "prepare_report.json").read_text())
            self.assertEqual(report["domains"]["cv"]["mismatches"], 12)
            summary = self.run_prepare(root, sources, "--max-mismatch-rate", "1.0")
            self.assertEqual(summary["domains"]["uav"]["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
