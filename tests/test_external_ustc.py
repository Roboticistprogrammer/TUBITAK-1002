import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from firecls.data.ustc import USTC_CLASSES, UstcDataset, binary_label

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_external.py"
spec = importlib.util.spec_from_file_location("evaluate_external", SCRIPT)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


class UstcDatasetTests(unittest.TestCase):
    def test_binary_label_is_smoke_only(self):
        self.assertEqual([binary_label(c) for c in USTC_CLASSES], [1, 0, 0, 0, 0, 0])

    def test_loads_index_and_converts_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "datasets" / "USTC_SmokeRS" / "smoke"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (256, 256), (10, 20, 30)).save(image_dir / "a.tif")
            index = root / "index.csv"
            with index.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["path", "label6", "label_bin", "split"])
                writer.writerow(["datasets/USTC_SmokeRS/smoke/a.tif", "smoke", 1, "test"])
            image, label6, label_bin, path = UstcDataset(index, root)[0]
            self.assertEqual((image.mode, image.size, label6, label_bin), ("RGB", (256, 256), 0, 1))
            self.assertTrue(path.endswith("a.tif"))

    def test_images_dir_overrides_index_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clone" / "smoke").mkdir(parents=True)
            Image.new("RGB", (256, 256), (1, 2, 3)).save(root / "clone" / "smoke" / "a.tif")
            index = root / "index.csv"
            index.write_text("path,label6,label_bin,split\ndatasets/USTC_SmokeRS/smoke/a.tif,smoke,1,test\n", encoding="utf-8")
            dataset = UstcDataset(index, root / "elsewhere", images_dir=root / "clone")
            self.assertEqual(dataset[0][0].size, (256, 256))

    def test_rejects_inconsistent_binary_label(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.csv"
            index.write_text("path,label6,label_bin,split\nx.tif,cloud,1,test\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                UstcDataset(index, Path(directory))


class ExternalMetricTests(unittest.TestCase):
    def test_auc_matches_hand_computation(self):
        score = np.array([0.1, 0.4, 0.35, 0.8])
        positive = np.array([False, False, True, True])
        self.assertAlmostEqual(ext.roc_auc(score, positive), 0.75)  # sklearn docs example
        self.assertAlmostEqual(ext.roc_auc(np.array([0.5, 0.5]), np.array([True, False])), 0.5)  # ties

    def test_mcnemar_exact(self):
        self.assertAlmostEqual(ext.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(ext.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(ext.mcnemar_exact(0, 10), 2 / 1024)
        self.assertAlmostEqual(ext.mcnemar_exact(2, 8), 2 * (1 + 10 + 45) / 1024)

    def test_holm(self):
        adjusted = ext.holm([0.01, 0.04, 0.03])
        self.assertTrue(np.allclose(adjusted, [0.03, 0.06, 0.06]))

    def test_core_metrics(self):
        pos_pred = np.array([True, True, False, True, False, False])
        labels = np.array([True, True, True, False, False, False])
        score = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])
        m = ext.core_metrics(pos_pred, score, labels)
        self.assertAlmostEqual(m["smoke_recall"], 2 / 3)
        self.assertAlmostEqual(m["false_alarm_rate"], 1 / 3)
        self.assertAlmostEqual(m["balanced_accuracy"], 2 / 3)

    def test_prediction_rule_neither_is_negative(self):
        logits = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "m"
            folder.mkdir()
            np.savez(folder / "ustc.npz", logits=logits, labels6=np.array([0, 0, 1, 1]),
                     labels_bin=np.array([1, 1, 0, 0]), names=np.array(list("abcd")),
                     classes=np.array(ext.CLASSES), classes6=np.array(ext.USTC_CLASSES))
            run = ext.load(folder)
        primary, secondary = ext.apply_rule(run, "primary"), ext.apply_rule(run, "secondary")
        self.assertEqual(primary["pos_pred"].tolist(), [True, True, True, False])  # fire, smoke, both -> positive
        self.assertEqual(secondary["pos_pred"].tolist(), [False, True, True, False])  # fire counts as negative
        self.assertTrue(np.all(primary["score"][:3] > primary["score"][3]))
        self.assertLess(secondary["score"][0], secondary["score"][1])  # fire-only is not a smoke score

    def test_full_resize_transform_keeps_whole_image(self):
        from firecls.transforms import build_eval_transform, build_full_resize_transform

        mean, std = [0.0] * 3, [1.0] * 3
        image = Image.new("RGB", (256, 256), (0, 0, 0))
        image.paste((255, 255, 255), (0, 0, 16, 256))  # a bright strip at the left border, outside the centre crop
        full = build_full_resize_transform(192, mean, std)(image)
        crop = build_eval_transform(192, mean, std)(image)
        self.assertEqual(tuple(full.shape), (3, 192, 192))
        self.assertGreater(float(full[:, :, :8].mean()), 0.5)  # the border strip survives the full resize
        self.assertLess(float(crop.max()), 0.05)  # the 224 resize + 192 crop discards it


if __name__ == "__main__":
    unittest.main()
