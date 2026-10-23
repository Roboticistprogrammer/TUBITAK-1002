import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from firecls import binary_eval as be
from firecls.data.plumes import DOMAIN_ORDER, RS_DOMAIN_ID, PlumeDataset
from firecls.distillation.losses import MultiTeacherDistillationLoss
from firecls.finetune.data import build_train_dataset, build_val_datasets, rs_class_balanced_sampler
from firecls.finetune.recipes import ORIGINAL, finetune_recipe, lr_factor_at
from firecls.finetune.runner import make_optimizer, run_finetune, train_supervised_epoch

CLASSES = ["fire", "smoke", "both", "neither"]


def write_images_and_csv(csv_path: Path, image_dir: Path, rows: list[tuple[str, str, str]], extra: dict | None = None) -> None:
    """rows: (file name, label, split); writes tiny PNGs and a CSV with absolute paths."""
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "split"] + (list(extra) if extra else [])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, label, split in rows:
            Image.new("RGB", (16, 16), (len(name) * 7 % 255, 20, 30)).save(image_dir / name)
            writer.writerow({"path": str(image_dir / name), "label": label, "split": split, **(extra or {})})


def make_world(root: Path) -> None:
    """A miniature dataset root: FASDD cv/rs/uav (train + val) and the plume index with train + val rows."""
    layout = {"cv": "datasets/FASDD_CV/FASDD_CV/images", "rs": "datasets/FASDD_RS/images", "uav": "datasets/FASDD_UAV/images"}
    for key, folder in layout.items():
        rows = [(f"{key}_{i}.png", CLASSES[i % 4], "train" if i < 8 else "val") for i in range(12)]
        write_images_and_csv(root / "data_index" / f"fasdd_{key}_tdml.csv", root / folder, rows)
    plume_dir = root / "datasets" / "IndustrialSmokePlumes" / "rgb"
    with (root / "data_index" / "industrial_plumes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "site", "group", "month", "label_src", "label", "split"])
        for i in range(6):
            src, label = ("positive", "smoke") if i % 2 == 0 else ("negative", "neither")
            rel = f"datasets/IndustrialSmokePlumes/rgb/{src}/p{i}.png"
            (plume_dir / src).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (200, 10 * i, 10)).save(root / rel)
            writer.writerow([rel, i, i, 1, src, label, "train" if i < 4 else "val"])


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(x.flatten(1))


class BinaryMetricTests(unittest.TestCase):
    def test_metrics_hand_computed(self):
        truth = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        pred = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
        score = np.array([.9, .8, .7, .4, .6, .3, .2, .1])
        m = be.binary_metrics(pred, score, truth)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"], m["errors"]), (3, 1, 1, 3, 2))
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertAlmostEqual(m["false_alarm_rate"], 0.25)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(m["f1"], 0.75)
        self.assertAlmostEqual(m["roc_auc"], 15 / 16)

    def test_prediction_rule_and_score(self):
        logits = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], dtype=float)
        self.assertEqual(be.positive_prediction(logits).tolist(), [True, True, True, False])  # fire, smoke, both -> positive
        self.assertTrue(be.positive_score(logits)[0] > 0.9 and be.positive_score(logits)[3] < 0.1)
        self.assertEqual(be.positive_truth(np.array([0, 1, 2, 3])).tolist(), [True, True, True, False])

    def test_recall_on_subclass(self):
        labels = np.array([1, 1, 2, 2, 2, 3])
        pred = np.array([True, False, True, True, False, False])
        self.assertEqual(be.recall_on(pred, labels, be.SMOKE), {"images": 2, "recall": 0.5})
        self.assertAlmostEqual(be.recall_on(pred, labels, be.BOTH)["recall"], 2 / 3)

    def test_mcnemar_and_holm(self):
        self.assertAlmostEqual(be.mcnemar_exact(0, 10), 2 / 1024)
        self.assertAlmostEqual(be.mcnemar_exact(5, 5), 1.0)
        self.assertTrue(np.allclose(be.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
        truth = np.array([1, 1, 0, 0], dtype=bool)
        result = be.paired_test(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool), truth)
        self.assertEqual((result["a_right_b_wrong"], result["a_wrong_b_right"]), (2, 0))


class PlumeDomainTests(unittest.TestCase):
    def test_plume_dataset_rejects_label_folder_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "i.csv"
            index.write_text("path,site,group,month,label_src,label,split\nx.png,1,1,1,negative,smoke,train\n")
            with self.assertRaises(ValueError):
                PlumeDataset(index, Path(tmp), "train", CLASSES)

    def test_plume_samples_get_rs_domain_id_and_rs_teacher_soft_target(self):
        """Required by the protocol: a plume sample is an RS-domain sample and, under routed aggregation, its soft target
        is the RS teacher's, not the CV or UAV teacher's."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_world(root)
            tf = lambda image: torch.zeros(3, 8, 8) + float(image.getpixel((0, 0))[0])  # noqa: E731
            dataset, sizes = build_train_dataset(["cv", "rs", "uav"], CLASSES, tf, root, root, include_plumes=True, tag_domains=True)
            self.assertEqual(list(sizes), ["fasdd_cv", "fasdd_rs", "plume", "fasdd_uav"])
            self.assertEqual(sizes["plume"], 4)
            plume_start = sizes["fasdd_cv"] + sizes["fasdd_rs"]
            image, label, domain = dataset[plume_start]
            self.assertEqual(domain, RS_DOMAIN_ID)
            self.assertEqual(DOMAIN_ORDER[domain], "rs")
            self.assertEqual({dataset[plume_start + i][2] for i in range(4)}, {RS_DOMAIN_ID})
            self.assertEqual(dataset[0][2], DOMAIN_ORDER.index("cv"))
            self.assertEqual(dataset[len(dataset) - 1][2], DOMAIN_ORDER.index("uav"))
            # routed aggregation over a batch mixing cv, rs, plume and uav samples
            batch = [dataset[0], dataset[sizes["fasdd_cv"]], dataset[plume_start], dataset[len(dataset) - 1]]
            domain_ids = torch.tensor([b[2] for b in batch])
            torch.manual_seed(0)
            teacher_logits = [torch.randn(4, 4) for _ in range(3)]  # order cv, rs, uav
            loss = MultiTeacherDistillationLoss(3, temperature=4.0, aggregation="domain_routed")
            target = loss._aggregate_teacher_soft(teacher_logits, domain_ids)
            expected_rs = F.softmax(teacher_logits[1][2] / 4.0, dim=0)  # sample 2 is the plume image
            self.assertTrue(torch.allclose(target[2], expected_rs, atol=1e-6))
            self.assertFalse(torch.allclose(target[2], F.softmax(teacher_logits[0][2] / 4.0, dim=0), atol=1e-6))

    def test_control_has_no_plume_and_untagged_samples_are_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_world(root)
            tf = lambda image: torch.zeros(3, 8, 8)  # noqa: E731
            control, sizes = build_train_dataset(["cv", "rs", "uav"], CLASSES, tf, root, root, include_plumes=False, tag_domains=False)
            self.assertNotIn("plume", sizes)
            self.assertEqual(len(control[0]), 2)
            teacher_only, sizes = build_train_dataset(["rs"], CLASSES, tf, root, root, include_plumes=True, tag_domains=False)
            self.assertEqual(list(sizes), ["fasdd_rs", "plume"])
            vals = build_val_datasets(["rs"], CLASSES, tf, root, root)
            self.assertEqual(sorted(vals), ["rs_fasdd", "rs_plume"])
            self.assertEqual(sorted(build_val_datasets(["cv", "rs", "uav"], CLASSES, tf, root, root)),
                             ["cv", "rs_fasdd", "rs_plume", "uav"])

    def test_balanced_sampler_touches_only_the_rs_portion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_world(root)
            tf = lambda image: torch.zeros(3, 8, 8)  # noqa: E731
            dataset, sizes = build_train_dataset(["cv", "rs", "uav"], CLASSES, tf, root, root, include_plumes=True, tag_domains=False)
            sampler = rs_class_balanced_sampler(dataset, list(sizes), seed=0)
            weights = np.array(sampler.weights)
            n_cv, n_rs, n_plume = sizes["fasdd_cv"], sizes["fasdd_rs"], sizes["plume"]
            self.assertTrue(np.allclose(weights[:n_cv], 1.0))
            self.assertTrue(np.allclose(weights[n_cv + n_rs + n_plume:], 1.0))
            rs_weights = weights[n_cv:n_cv + n_rs + n_plume]
            self.assertAlmostEqual(float(rs_weights.sum()), n_rs + n_plume)  # RS portion keeps its total weight
            labels = [lab for part in dataset.datasets[1:3] for _, lab in part.samples]
            totals = {lab: rs_weights[[i for i, l in enumerate(labels) if l == lab]].sum() for lab in set(labels)}
            self.assertTrue(np.allclose(list(totals.values()), list(totals.values())[0]))  # equal mass per class
            self.assertEqual(sampler.num_samples, len(dataset))


class SourceBalancedBatchTests(unittest.TestCase):
    def test_every_batch_is_half_fasdd_rs_half_plume(self):
        """RS-teacher fallback: each batch has exactly 50% FASDD-RS and 50% plume images."""
        from firecls.finetune.data import source_balanced_batch_sampler

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_world(root)
            tf = lambda image: torch.zeros(3, 8, 8)  # noqa: E731
            dataset, sizes = build_train_dataset(["rs"], CLASSES, tf, root, root, include_plumes=True, tag_domains=False)
            self.assertEqual(list(sizes), ["fasdd_rs", "plume"])
            n_rs, n_plume = sizes["fasdd_rs"], sizes["plume"]  # 8 and 4: plumes are the minority, so they are reused
            sampler = source_balanced_batch_sampler(dataset, list(sizes), [["fasdd_rs"], ["plume"]], batch_size=4, seed=7)
            self.assertEqual(len(sampler), (n_rs + n_plume) // 4)
            seen_plume = set()
            for _ in range(3):  # three epochs: the queues continue, every batch stays balanced
                for batch in sampler:
                    self.assertEqual(len(batch), 4)
                    self.assertEqual(sum(i < n_rs for i in batch), 2)
                    self.assertEqual(sum(i >= n_rs for i in batch), 2)
                    self.assertEqual(len(set(batch)), 4)  # no duplicate inside a batch
                    seen_plume.update(i for i in batch if i >= n_rs)
            self.assertEqual(seen_plume, set(range(n_rs, n_rs + n_plume)))  # every plume image is reached
            again = source_balanced_batch_sampler(dataset, list(sizes), [["fasdd_rs"], ["plume"]], batch_size=4, seed=7)
            first = source_balanced_batch_sampler(dataset, list(sizes), [["fasdd_rs"], ["plume"]], batch_size=4, seed=7)
            self.assertEqual(list(again), list(first))  # deterministic for a seed
            loader = DataLoader(dataset, batch_sampler=first)
            labels_per_batch = [len(images) for images, *_ in loader]
            self.assertEqual(labels_per_batch, [4, 4, 4])

    def test_batch_size_must_divide_by_sources(self):
        from firecls.finetune.data import SourceBalancedBatchSampler

        with self.assertRaises(ValueError):
            SourceBalancedBatchSampler([[0, 1], [2, 3]], batch_size=5, num_batches=1, seed=0)


class RecipeTests(unittest.TestCase):
    def test_finetune_recipes_follow_the_protocol(self):
        for kind, base_key, batch in [("kd_routed", "kd", 8), ("kd_uniform", "kd", 8), ("nokd", "nokd", 8),
                                      ("mobilenet_v3_large", "cnn", 16), ("resnet50", "cnn", 16), ("teacher", "teacher", 8)]:
            r = finetune_recipe(kind)
            self.assertAlmostEqual(r.lr, ORIGINAL[base_key].lr * 0.1)
            self.assertEqual(r.batch_size, batch)
            self.assertEqual(r.amp, ORIGINAL[base_key].amp)
        self.assertFalse(finetune_recipe("kd_routed").amp)  # students trained without AMP; the no-KD student with it
        self.assertTrue(finetune_recipe("nokd").amp)
        self.assertEqual(finetune_recipe("kd_routed", batch_size=4).batch_size, 4)

    def test_schedule_restarts_over_the_finetuning(self):
        cnn = finetune_recipe("resnet50")
        steps = 10
        self.assertLess(lr_factor_at(cnn, 0, steps, 5), 0.2)          # warm-up starts again from a small value
        self.assertAlmostEqual(lr_factor_at(cnn, steps, steps, 5), 1.0)  # end of warm-up epoch
        self.assertLess(lr_factor_at(cnn, 5 * steps - 1, steps, 5), 0.01)  # cosine reaches ~0 after the 5th epoch
        self.assertEqual(lr_factor_at(finetune_recipe("kd_routed"), 123, steps, 5), 1.0)  # constant for the students


class FinetuneRunnerTests(unittest.TestCase):
    def test_two_epoch_run_selects_best_and_keeps_checkpoint_format(self):
        torch.manual_seed(0)

        def loader(n, seed):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(n, 3, 8, 8, generator=g)
            y = torch.randint(0, 4, (n,), generator=g)
            return DataLoader(TensorDataset(x, y), batch_size=8)

        model = TinyNet()
        base = {"model": model.state_dict(), "classes": CLASSES, "model_name": "tiny", "optimizer": {"x": 1}, "avg_val": 0.1, "seed": 42}
        val = {"cv": loader(16, 1), "uav": loader(16, 2), "rs_fasdd": loader(16, 3), "rs_plume": loader(16, 4)}
        train = loader(32, 5)
        recipe = finetune_recipe("resnet50")
        optimizer, scheduler = make_optimizer(model.parameters(), recipe, len(train), 2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = run_finetune(
                model=model, base_checkpoint=base, val_loaders=val, recipe=recipe, epochs=2, device=torch.device("cpu"),
                output_dir=out, metadata={"kind": "test"},
                train_epoch=lambda e: train_supervised_epoch(model, train, optimizer, scheduler, scaler, recipe, torch.device("cpu"), f"e{e}"),
            )
            import json
            history = json.loads((out / "history.json").read_text())["records"]
            self.assertEqual([r["epoch"] for r in history], [0, 1, 2])  # epoch 0 = before, never selected
            record = history[1]
            expected = np.mean([record["cv_acc"], record["uav_acc"], record["rs_pooled_binary_ba"]])
            self.assertAlmostEqual(record["selection_score"], float(expected))
            self.assertIn("rs_fasdd", record)
            self.assertIn("rs_plume", record)
            self.assertEqual(sorted(record["pred_share"]), ["cv", "rs_fasdd", "rs_plume", "uav"])
            self.assertAlmostEqual(sum(record["pred_share"]["rs_plume"]), 1.0)
            self.assertTrue(len(record["loss_by_tenth_of_epoch"]) >= 1)
            # RS-teacher fallback: FASDD-RS val binary balanced accuracy drop of the selected checkpoint against epoch 0
            best_ba = history[summary["best_epoch"]]["rs_fasdd"]["balanced_accuracy"]
            self.assertAlmostEqual(summary["fasdd_rs_val_binary_ba_best"], best_ba)
            self.assertAlmostEqual(summary["fasdd_rs_val_ba_drop_points"], 100 * (history[0]["rs_fasdd"]["balanced_accuracy"] - best_ba))
            self.assertEqual(summary["ba_fallback_triggered"], summary["fasdd_rs_val_ba_drop_points"] > 2.0)
            self.assertIn(summary["best_epoch"], (1, 2))
            self.assertEqual(summary["best_selection_score"], max(history[1]["selection_score"], history[2]["selection_score"]))
            # the KD loaders (load_teacher_checkpoint) use torch.load's default safe mode: only plain types may be saved
            saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
            torch.load(out / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["classes"], CLASSES)
            self.assertEqual(saved["model_name"], "tiny")
            self.assertNotIn("optimizer", saved)
            self.assertEqual(saved["finetune"]["kind"], "test")
            self.assertEqual(saved["epoch"], summary["best_epoch"])
            TinyNet().load_state_dict(saved["model"])
        self.assertIsInstance(summary["fallback_triggered"], bool)


if __name__ == "__main__":
    unittest.main()
