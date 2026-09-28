import itertools
import unittest

import numpy as np
import torch

from firecls.baselines.power import parse_power_mw
from firecls.baselines.preprocessing import Letterbox, build_transform, resize_center_crop_spec
from firecls.baselines.protocol import CLASSES
from firecls.baselines.scores import PresenceToScores, calibrate_thresholds, presence_to_scores


def thresholded_label(p_fire, p_smoke, t_fire, t_smoke):
    fire, smoke = p_fire >= t_fire, p_smoke >= t_smoke
    return {(True, False): "fire", (False, True): "smoke", (True, True): "both", (False, False): "neither"}[
        (bool(fire), bool(smoke))
    ]


class PresenceScoreTests(unittest.TestCase):
    def test_argmax_equals_thresholded_decision(self):
        rng = np.random.default_rng(0)
        p_fire, p_smoke = rng.random(2000), rng.random(2000)
        for t_fire, t_smoke in itertools.product([0.1, 0.35, 0.5, 0.8], repeat=2):
            scores = presence_to_scores(p_fire, p_smoke, t_fire, t_smoke)
            predicted = [CLASSES[i] for i in scores.argmax(1)]
            expected = [thresholded_label(f, s, t_fire, t_smoke) for f, s in zip(p_fire, p_smoke)]
            self.assertEqual(predicted, expected)

    def test_torch_module_matches_numpy(self):
        p_fire, p_smoke = torch.rand(64), torch.rand(64)
        head = PresenceToScores(0.3, 0.7)
        np.testing.assert_allclose(
            head(p_fire, p_smoke).numpy(), presence_to_scores(p_fire.numpy(), p_smoke.numpy(), 0.3, 0.7), rtol=1e-5
        )

    def test_calibration_recovers_generating_thresholds(self):
        rng = np.random.default_rng(1)
        p_fire, p_smoke = rng.random(3000), rng.random(3000)
        labels = np.array(
            [CLASSES.index(thresholded_label(f, s, 0.3, 0.6)) for f, s in zip(p_fire, p_smoke)], dtype=np.int64
        )
        best = calibrate_thresholds({"cv": (labels, p_fire, p_smoke)})
        self.assertAlmostEqual(best["t_fire"], 0.3)
        self.assertAlmostEqual(best["t_smoke"], 0.6)
        self.assertAlmostEqual(best["objective"], 1.0)


class PreprocessingTests(unittest.TestCase):
    def test_letterbox_shape_and_padding(self):
        from PIL import Image

        tensor = Letterbox(64)(Image.new("RGB", (128, 32), (255, 0, 0)))
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 114 / 255, places=5)  # padded row
        self.assertAlmostEqual(float(tensor[0, 32, 32]), 1.0, places=5)  # image content

    def test_resize_center_crop(self):
        from PIL import Image

        out = build_transform(resize_center_crop_spec(192))(Image.new("RGB", (640, 480)))
        self.assertEqual(tuple(out.shape), (3, 192, 192))


class PowerParsingTests(unittest.TestCase):
    def test_nano_and_orin_formats(self):
        nano = "RAM 2000/3964MB ... POM_5V_IN 4096/3500 POM_5V_GPU 1000/900 POM_5V_CPU 800/700"
        orin = "RAM 3000/15000MB ... VDD_IN 7200mW/6900mW VDD_CPU_GPU_CV 2100mW/2000mW VDD_SOC 1500mW/1400mW"
        self.assertEqual(parse_power_mw(nano), ("POM_5V_IN", 4096.0))
        self.assertEqual(parse_power_mw(orin), ("VDD_IN", 7200.0))
        self.assertIsNone(parse_power_mw("RAM 2000/3964MB CPU [10%@1400]"))


if __name__ == "__main__":
    unittest.main()


class BoxReaderTests(unittest.TestCase):
    def test_coco_and_yolo_agree(self):
        import json
        import tempfile
        from pathlib import Path

        from firecls.baselines.boxes import label_from_boxes, read_boxes

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            coco = {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
                           {"id": 2, "file_name": "sub/b.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 7, "name": "Smoke"}, {"id": 9, "name": "fire"}],
                "annotations": [{"image_id": 1, "category_id": 9, "bbox": [20, 10, 100, 50]},
                                {"image_id": 1, "category_id": 7, "bbox": [0, 0, 200, 100]}],
            }
            (d / "c.json").write_text(json.dumps(coco))
            labels = d / "labels"
            labels.mkdir()
            (labels / "a.txt").write_text("1 0.35 0.35 0.5 0.5\n0 0.5 0.5 1.0 1.0\n")
            (labels / "b.txt").write_text("")
            (d / "classes.txt").write_text("smoke\nfire\n")

            from_coco, from_yolo = read_boxes(d / "c.json"), read_boxes(labels)
            self.assertEqual(sorted(from_coco), ["a", "b"])
            self.assertEqual(label_from_boxes(from_coco["a"]), "both")
            self.assertEqual(label_from_boxes(from_yolo["b"]), "neither")
            fire_c = next(b for b in from_coco["a"] if b.cls == "fire")
            fire_y = next(b for b in from_yolo["a"] if b.cls == "fire")
            for u, v in zip((fire_c.x1, fire_c.y1, fire_c.x2, fire_c.y2), (fire_y.x1, fire_y.y1, fire_y.x2, fire_y.y2)):
                self.assertAlmostEqual(u, v, places=5)
