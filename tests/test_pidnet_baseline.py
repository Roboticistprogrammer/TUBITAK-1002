"""CPU-only tests for the Pesonen et al. PIDNet baseline (no dataset, no weight downloads)."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from firecls.baselines.boxes import Box
from firecls.baselines.pidnet.data import (
    JointAugment,
    PseudoMaskDataset,
    letterbox_geometry,
    letterbox_mask,
    mask_edges,
)
from firecls.baselines.pidnet.losses import PidnetLoss
from firecls.baselines.pidnet.masks import (
    decode_mask,
    encode_mask,
    generate_masks,
    mask_path,
    pseudo_mask,
    read_mask,
    write_mask,
)
from firecls.baselines.pidnet.metrics import SegmentationMeter
from firecls.baselines.pidnet.model import (
    PidnetPresenceScores,
    build_pidnet_s,
    load_imagenet_pretrained,
    presence_from_logits,
    topk_count,
)
from firecls.baselines.preprocessing import Letterbox
from firecls.baselines.protocol import CLASSES

SIZE = 128  # small input keeps CPU tests fast; PIDNet needs a multiple of 8


def thresholded(p_fire: float, p_smoke: float, t_fire: float, t_smoke: float) -> str:
    return {(True, False): "fire", (False, True): "smoke", (True, True): "both", (False, False): "neither"}[
        (p_fire >= t_fire, p_smoke >= t_smoke)
    ]


def save_checkpoint(directory: Path, net: torch.nn.Module, img_size: int = SIZE, topk_fraction: float = 0.01) -> Path:
    path = directory / "best.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "classes": list(CLASSES),
            "seg_channels": ["fire", "smoke"],
            "model_name": "pidnet_s",
            "img_size": img_size,
            "pad_value": 114,
            "topk_fraction": topk_fraction,
            "epoch": 3,
            "seed": 42,
            "select_by": "image_acc",
            "init": {"type": "random"},
            "history": [{"epoch": 1, "val": {"miou": None}}],
        },
        path,
    )
    return path


class ModelTests(unittest.TestCase):
    def test_forward_shapes_train_and_eval(self):
        torch.manual_seed(0)
        net = build_pidnet_s(2, augment=True)
        x = torch.rand(2, 3, SIZE, SIZE)
        outputs = net.train()(x)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[0].shape), (2, 2, SIZE // 8, SIZE // 8))  # P branch
        self.assertEqual(tuple(outputs[1].shape), (2, 2, SIZE // 8, SIZE // 8))  # main
        self.assertEqual(tuple(outputs[2].shape), (2, 1, SIZE // 8, SIZE // 8))  # D branch
        inference = build_pidnet_s(2, augment=False).eval()
        with torch.no_grad():
            self.assertEqual(tuple(inference(x).shape), (2, 2, SIZE // 8, SIZE // 8))

    def test_loss_backpropagates(self):
        torch.manual_seed(0)
        net = build_pidnet_s(2, augment=True).train()
        x = torch.rand(2, 3, SIZE, SIZE)
        mask = torch.zeros(2, 2, SIZE, SIZE)
        mask[:, 0, 20:60, 20:60] = 1
        mask[:, 1, 40:100, 30:90] = 1
        edge = torch.from_numpy(np.stack([mask_edges(m.numpy().astype(np.uint8)) for m in mask]))
        loss, parts = PidnetLoss()(net(x), mask, edge)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"loss_p", "loss_boundary", "loss_main", "loss_bas"})
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_topk_count(self):
        self.assertEqual(topk_count(512, 0.001), 4)  # 64*64*0.001 = 4.1
        self.assertEqual(topk_count(128, 0.001), 1)
        with self.assertRaises(ValueError):
            topk_count(1024, 1.0)  # 16384 > TensorRT 8.x TopK limit
        with self.assertRaises(ValueError):
            topk_count(100, 0.01)  # not a multiple of 8

    def test_presence_is_topk_mean(self):
        logits = torch.full((1, 2, 4, 4), -20.0)
        logits[0, 0, 0, :2] = 20.0  # two confident fire pixels
        p_fire, p_smoke = presence_from_logits(logits, k=4)
        self.assertAlmostEqual(float(p_fire), 0.5, places=4)
        self.assertLess(float(p_smoke), 1e-6)

    def test_wrapper_scores_match_thresholded_presence(self):
        torch.manual_seed(0)
        net = build_pidnet_s(2, augment=False)
        for t_fire, t_smoke in [(0.5, 0.5), (0.2, 0.7)]:
            wrapper = PidnetPresenceScores(net, SIZE, 0.01, t_fire, t_smoke).eval()
            x = torch.rand(4, 3, SIZE, SIZE)
            with torch.no_grad():
                scores = wrapper(x)
                p_fire, p_smoke = wrapper.presence(x)
            self.assertEqual(tuple(scores.shape), (4, len(CLASSES)))
            for i in range(4):
                expected = thresholded(float(p_fire[i]), float(p_smoke[i]), t_fire, t_smoke)
                self.assertEqual(CLASSES[int(scores[i].argmax())], expected)

    def test_wrapper_accepts_training_network(self):
        net = build_pidnet_s(2, augment=True).eval()
        wrapper = PidnetPresenceScores(net, SIZE, 0.01)
        with torch.no_grad():
            self.assertEqual(tuple(wrapper(torch.rand(1, 3, SIZE, SIZE)).shape), (1, 4))

    def test_imagenet_loader_matches_by_name_and_shape(self):
        source = build_pidnet_s(1000, augment=True)  # heads differ in shape, backbone matches
        target = build_pidnet_s(2, augment=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "PIDNet_S_ImageNet.pth.tar"
            torch.save({"state_dict": {f"module.{k}": v for k, v in source.state_dict().items()}}, path)
            info = load_imagenet_pretrained(target, path)
        self.assertGreater(info["tensors_loaded"], 0)
        self.assertLess(info["tensors_loaded"], info["tensors_total"])
        self.assertTrue(torch.equal(target.conv1[0].weight, source.conv1[0].weight))


class ExportAndFamilyTests(unittest.TestCase):
    def test_onnx_export_verifies_with_topk(self):
        import onnx

        from firecls.baselines.export import export_scores_model
        from firecls.baselines.preprocessing import letterbox_spec

        torch.manual_seed(0)
        net = build_pidnet_s(2, augment=False)
        wrapper = PidnetPresenceScores(net, SIZE, 0.05, 0.4, 0.6).eval()
        self.assertGreater(wrapper.k, 1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            checkpoint = d / "best.pt"
            checkpoint.write_bytes(b"placeholder")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = export_scores_model(
                    wrapper.export_module(), d / "pidnet.onnx", letterbox_spec(SIZE), checkpoint,
                    family="pidnet", model_name="pidnet_s", source="reimplemented", opset=17,
                )
            graph = onnx.load(path).graph
            self.assertIn("TopK", {node.op_type for node in graph.node})
            manifest = json.loads(path.with_suffix(".onnx.manifest.json").read_text())
            self.assertTrue(manifest["verification"]["top1_match"])
            self.assertEqual(manifest["opset"], 17)

    def test_family_loader_round_trip(self):
        from firecls.baselines.pidnet.checkpoint import write_thresholds
        from firecls.baselines.registry import available_families, load_baseline

        self.assertIn("pidnet", available_families())
        torch.manual_seed(0)
        net = build_pidnet_s(2, augment=True)  # training net with auxiliary heads
        with tempfile.TemporaryDirectory() as d:
            checkpoint = save_checkpoint(Path(d), net)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_baseline("pidnet", checkpoint)
            self.assertTrue(any("uncalibrated" in str(w.message) for w in caught))
            self.assertFalse(loaded.metadata["thresholds_calibrated"])
            self.assertEqual(loaded.metadata["thresholds"], {"t_fire": 0.5, "t_smoke": 0.5})
            self.assertEqual(loaded.metadata["source"], "reimplemented")
            self.assertEqual(loaded.metadata["classes"], CLASSES)
            self.assertEqual(loaded.preprocessing, {"mode": "letterbox", "size": SIZE, "pad_value": 114, "scale": "1/255"})

            x = torch.rand(2, 3, SIZE, SIZE)
            reference = PidnetPresenceScores(net.eval(), SIZE, 0.01).eval()
            with torch.no_grad():
                torch.testing.assert_close(loaded.model(x), reference(x))

            write_thresholds(checkpoint, {"t_fire": 0.3, "t_smoke": 0.65})
            calibrated = load_baseline("pidnet", checkpoint)
            self.assertTrue(calibrated.metadata["thresholds_calibrated"])
            self.assertEqual(calibrated.metadata["thresholds"], {"t_fire": 0.3, "t_smoke": 0.65})
            self.assertAlmostEqual(calibrated.model.head.t_fire, 0.3)

            # A re-trained checkpoint must not silently reuse stale thresholds.
            save_checkpoint(Path(d), net, topk_fraction=0.02)
            with self.assertRaises(ValueError):
                load_baseline("pidnet", checkpoint)


class MaskTests(unittest.TestCase):
    def test_bit_encoding_round_trip(self):
        rng = np.random.default_rng(0)
        fire, smoke = rng.random((2, 37, 53)) > 0.5
        encoded = encode_mask(fire, smoke)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertTrue(set(np.unique(encoded).tolist()) <= {0, 1, 2, 3})
        with tempfile.TemporaryDirectory() as d:
            path = mask_path(Path(d), "uav", r"E:\x\images\bothFireAndSmoke_UAV000001.jpg")
            self.assertEqual(path, Path(d) / "uav" / "bothFireAndSmoke_UAV000001.png")
            write_mask(path, encoded)
            decoded = decode_mask(read_mask(path))
        np.testing.assert_array_equal(decoded[0], fire)
        np.testing.assert_array_equal(decoded[1], smoke)

    def test_pseudo_mask_with_mock_sam_and_fallbacks(self):
        class FakeSam:
            def __init__(self):
                self.calls = 0

            def set_image(self, image):
                self.shape = image.shape[:2]

            def predict_box(self, box):
                self.calls += 1
                mask = np.zeros(self.shape, dtype=bool)
                if self.calls == 1:  # a proper segment inside the first box
                    x1, y1, x2, y2 = box.astype(int)
                    mask[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = True
                    return mask
                if self.calls == 2:
                    return mask  # empty -> filled-box fallback
                raise RuntimeError("SAM failure")  # -> filled-box fallback

        image = np.zeros((40, 80, 3), dtype=np.uint8)
        boxes = [Box("fire", 0.1, 0.1, 0.4, 0.6), Box("smoke", 0.5, 0.0, 1.0, 0.5), Box("smoke", 0.0, 0.5, 0.2, 1.0)]
        encoded, stats = pseudo_mask(FakeSam(), image, boxes)
        fire, smoke = decode_mask(encoded)
        self.assertEqual(stats, {"boxes": 3, "fallback_error": 1, "fallback_empty": 1})
        self.assertTrue(fire[10, 16] and not fire[4, 8])  # SAM segment, shrunk inside the box
        self.assertTrue(smoke[0:20, 40:80].all())  # filled box (empty SAM mask)
        self.assertTrue(smoke[20:40, 0:16].all())  # filled box (SAM error)
        self.assertFalse(np.any(encoded[25:40, 40:80]))

        empty, stats = pseudo_mask(FakeSam(), image, [])
        self.assertFalse(empty.any())
        self.assertEqual(stats["boxes"], 0)

    def test_generate_masks_is_resumable(self):
        class BoxSam:
            def set_image(self, image):
                self.shape = image.shape[:2]

            def predict_box(self, box):
                mask = np.zeros(self.shape, dtype=bool)
                x1, y1, x2, y2 = box.astype(int)
                mask[y1:y2, x1:x2] = True
                return mask

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            paths = []
            for name in ["fire_a", "neither_b", "smoke_c"]:
                paths.append(d / f"{name}.jpg")
                Image.new("RGB", (32, 24)).save(paths[-1])
            samples = {"cv": [(paths[0], "fire"), (paths[1], "neither"), (paths[2], "smoke")]}
            boxes = {"cv": {"fire_a": [Box("fire", 0, 0, 0.5, 0.5)], "neither_b": []}}  # smoke_c has no entry
            stats = generate_masks(samples, boxes, BoxSam(), d / "masks", limit=2)
            self.assertEqual(stats["cv"]["written"], 2)
            stats = generate_masks(samples, boxes, BoxSam(), d / "masks")
            self.assertEqual((stats["cv"]["written"], stats["cv"]["skipped_existing"]), (1, 2))
            self.assertEqual(stats["cv"]["missing_box_entry"], 1)
            self.assertEqual(stats["cv"]["label_mismatch"], 1)  # smoke_c labelled smoke, no boxes
            fire, smoke = decode_mask(read_mask(d / "masks" / "cv" / "fire_a.png"))
            self.assertTrue(fire[:12, :16].all() and not fire[12:, :].any() and not smoke.any())


class DataTests(unittest.TestCase):
    def _square_sample(self, directory: Path, size=(160, 96)):
        """Image with a pure red rectangle whose fire mask marks exactly that rectangle."""
        width, height = size
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[20:60, 50:110] = (255, 0, 0)
        image_path = directory / "fire_sq.png"
        Image.fromarray(array).save(image_path)
        fire = np.zeros((height, width), dtype=bool)
        fire[20:60, 50:110] = True
        write_mask(mask_path(directory / "masks", "cv", image_path), encode_mask(fire, np.zeros_like(fire)))
        return image_path

    def test_letterbox_mask_matches_image_letterbox(self):
        self.assertEqual(letterbox_geometry(160, 96, 64), (64, 38, 0, 13))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            image_path = self._square_sample(d)
            dataset = PseudoMaskDataset([(image_path, 0)], "cv", d / "masks", 64, augment=False)
            image, mask, edge, label = dataset[0]
            reference = Letterbox(64)(Image.open(image_path).convert("RGB"))
            torch.testing.assert_close(image, reference)  # identical to the evaluation transform
            self.assertEqual(tuple(mask.shape), (2, 64, 64))
            self.assertEqual(tuple(edge.shape), (64, 64))
            self.assertEqual(label, 0)
            red = (image[0] > 0.9) & (image[1] < 0.1)
            self.assertGreater(int(red.sum()), 0)
            iou = float((red & (mask[0] > 0)).sum()) / float((red | (mask[0] > 0)).sum())
            self.assertGreater(iou, 0.85)
            self.assertGreater(float(edge.sum()), 0)

    def test_joint_augmentation_keeps_alignment(self):
        torch.manual_seed(0)
        size = 64
        image = torch.zeros(3, size, size)
        image[0, 16:40, 20:44] = 1.0  # red square on black
        mask = torch.zeros(2, size, size, dtype=torch.uint8)
        mask[0, 16:40, 20:44] = 1
        augment = JointAugment(size)
        geometric = {"crop", "vflip", "rotation", "perspective", "erasing"}
        for index, name in enumerate(augment.names):
            if name not in geometric:
                continue
            for _ in range(5):
                out_image, out_mask = augment(image.clone(), mask.clone(), op=index)
                self.assertEqual(tuple(out_image.shape), (3, size, size))
                self.assertEqual(out_mask.dtype, torch.uint8)
                self.assertTrue(set(out_mask.unique().tolist()) <= {0, 1}, name)
                red = (out_image[0] > 0.5) & (out_image[1] < 0.3)
                fire = out_mask[0] > 0
                union = int((red | fire).sum())
                if union == 0:
                    continue  # e.g. crop or erasing removed the square entirely
                self.assertGreater(float((red & fire).sum()) / union, 0.8, name)
                self.assertFalse(out_mask[1].any())
        for index, name in enumerate(augment.names):  # photometric ops never touch the mask
            if name in geometric:
                continue
            for _ in range(8):  # covers both outcomes of the random horizontal flip
                out_image, out_mask = augment(image.clone(), mask.clone(), op=index)
                self.assertEqual(tuple(out_image.shape), (3, size, size))
                self.assertTrue(torch.equal(out_mask, mask) or torch.equal(out_mask, mask.flip(-1)), name)


class MetricTests(unittest.TestCase):
    def test_sample_wise_iou(self):
        meter = SegmentationMeter()
        mask = torch.zeros(2, 2, 4, 4)
        mask[0, 0, :2, :2] = 1  # sample 0: 4 fire pixels
        logits = torch.full((2, 2, 4, 4), -10.0)
        logits[0, 0, :2, :] = 10.0  # predicts 8 pixels -> IoU 0.5
        logits[1, 1, 0, 0] = 10.0  # false positive smoke on an image without smoke
        meter.update(logits, mask)
        result = meter.compute()
        self.assertAlmostEqual(result["iou_fire"], 0.5)
        self.assertIsNone(result["iou_smoke"])  # no image contains smoke
        self.assertEqual(result["pooled_iou_smoke"], 0.0)
        self.assertAlmostEqual(result["miou"], 0.5)


if __name__ == "__main__":
    unittest.main()
