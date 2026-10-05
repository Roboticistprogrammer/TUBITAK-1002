import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.distillation.losses import MultiTeacherDistillationLoss
from firecls.distillation.trainer import MultiTeacherDistillationTrainer

CLASSES = ["fire", "smoke", "both", "neither"]


def make_domain_csv(root: Path, name: str, count: int) -> Path:
    rows = []
    for i in range(count):
        image = root / f"{name}_{i}.png"
        Image.new("RGB", (16, 16), color=(i * 20 % 255, 10, 10)).save(image)
        rows.append({"path": str(image), "label": CLASSES[i % 4], "split": "train"})
    csv_path = root / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 16 * 16, 4)

    def forward(self, x):
        return self.fc(x.flatten(1))


class DomainRoutedLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.teacher_logits = [torch.randn(6, 4) for _ in range(3)]
        self.student_logits = torch.randn(6, 4)
        self.labels = torch.tensor([0, 1, 2, 3, 0, 1])
        self.domain_ids = torch.tensor([0, 1, 2, 0, 1, 2])

    def test_routed_target_is_in_domain_teacher_only(self):
        loss = MultiTeacherDistillationLoss(3, temperature=4.0, aggregation="domain_routed")
        target = loss._aggregate_teacher_soft(self.teacher_logits, self.domain_ids)
        for sample, domain in enumerate(self.domain_ids.tolist()):
            expected = F.softmax(self.teacher_logits[domain][sample] / 4.0, dim=0)
            self.assertTrue(torch.allclose(target[sample], expected, atol=1e-6))

    def test_routed_requires_domain_ids(self):
        loss = MultiTeacherDistillationLoss(3, aggregation="domain_routed")
        with self.assertRaises(ValueError):
            loss(self.student_logits, self.teacher_logits, self.labels)

    def test_weighted_avg_is_uniform_and_ignores_domain(self):
        loss = MultiTeacherDistillationLoss(3, temperature=4.0, aggregation="weighted_avg")
        with_ids = loss._aggregate_teacher_soft(self.teacher_logits, self.domain_ids)
        without = loss._aggregate_teacher_soft(self.teacher_logits)
        mean = torch.stack([F.softmax(t / 4.0, dim=1) for t in self.teacher_logits]).mean(0)
        self.assertTrue(torch.allclose(with_ids, without))
        self.assertTrue(torch.allclose(without, mean, atol=1e-6))


class DomainTaggedTrainingTests(unittest.TestCase):
    def test_end_to_end_plumbing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets = [
                ImageClassificationCSVDataset(
                    make_domain_csv(root, name, 8), "train", CLASSES, transform=lambda im: torch.zeros(3, 16, 16) + len(name),
                    domain_id=index,
                )
                for index, name in enumerate(["cv", "rs", "uav"])
            ]
            self.assertEqual(len(sets[1][0]), 3)  # image, label, domain id
            loader = DataLoader(ConcatDataset(sets), batch_size=8, shuffle=True)
            teachers = {"cv": TinyNet(), "rs": TinyNet(), "uav": TinyNet()}
            trainer = MultiTeacherDistillationTrainer(
                TinyNet(), teachers, aggregation="domain_routed", device=torch.device("cpu")
            )
            optimizer = torch.optim.SGD(trainer.student.parameters(), lr=0.01)
            metrics = trainer.train_epoch(loader, optimizer, 1)
            self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))

    def test_untagged_dataset_still_returns_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = ImageClassificationCSVDataset(
                make_domain_csv(Path(tmp), "cv", 4), "train", CLASSES, transform=lambda im: torch.zeros(3, 16, 16)
            )
            self.assertEqual(len(dataset[0]), 2)


if __name__ == "__main__":
    unittest.main()
