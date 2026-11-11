"""USTC_SmokeRS external-validation dataset (see external/USTC_PROTOCOL.md).

The index `data_index/ustc_smokers.csv` has columns `path,label6,label_bin,split`. `path` is relative to the directory
that contains `datasets/`, in POSIX form (`datasets/USTC_SmokeRS/<class>/<file>.tif`), so the same index works on
Windows, Linux and the Jetson. Every image is `split=test`: nothing is trained or tuned on this dataset.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

USTC_CLASSES: list[str] = ["smoke", "cloud", "dust", "haze", "land", "seaside"]
USTC_POSITIVE = "smoke"  # every other class is a negative (protocol, section 5)
USTC_DIR = Path("datasets") / "USTC_SmokeRS"
USTC_INDEX = Path("data_index") / "ustc_smokers.csv"
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def binary_label(class_name: str) -> int:
    return int(class_name == USTC_POSITIVE)


class UstcDataset(Dataset):
    """Returns (image, label6, label_bin, path). Images are opened exactly as FASDD images are
    (`PIL.Image.open(path).convert("RGB")`), then pass through the shared eval transform."""

    def __init__(self, index_csv: Path, dataset_root: Path, transform=None, split: str = "test",
                 images_dir: Path | None = None) -> None:
        """`images_dir`, if given, is the folder that holds `<class>/<file>.tif` (e.g. the `ustc_smokers/` folder of the
        Hugging Face test-dataset clone); the index's own path prefix is then ignored and only `<class>/<file>` is used."""
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.samples: list[tuple[Path, int, int]] = []
        with Path(index_csv).open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"].lower() != split.lower():
                    continue
                label6 = USTC_CLASSES.index(row["label6"].lower())
                label_bin = int(row["label_bin"])
                if label_bin != binary_label(USTC_CLASSES[label6]):
                    raise ValueError(f"{index_csv}: label_bin disagrees with label6 for {row['path']}")
                if images_dir is None:
                    path = self.dataset_root / row["path"]
                else:
                    path = Path(images_dir) / USTC_CLASSES[label6] / Path(row["path"]).name
                self.samples.append((path, label6, label_bin))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label6, label_bin = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label6, label_bin, str(path)
