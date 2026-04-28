from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class ImageClassificationCSVDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        split: str,
        class_names: List[str],
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = split
        self.class_names = class_names
        self.label_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.return_path = return_path
        self.samples: List[Tuple[Path, int]] = []

        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].lower() != split.lower():
                    continue
                label = row["label"].lower()
                if label not in self.label_to_id:
                    raise ValueError(f"Unknown label '{label}' in {csv_path}")
                self.samples.append((Path(row["path"]), self.label_to_id[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(image_path)
        return image, label
