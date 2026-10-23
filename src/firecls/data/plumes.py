"""Industrial Smoke Plumes images as an extra remote-sensing (RS) domain (external/PLUME_PROTOCOL.md).

`data_index/industrial_plumes.csv` has the columns `path,site,group,month,label_src,label,split`. `path` is relative to the
directory that contains `datasets/` (the converted 8-bit RGB PNG), `label` is the four-class training label (`smoke` for
plume images, `neither` for non-plume images). Samples are returned in the same tuple layout as
`ImageClassificationCSVDataset`: (image, label[, domain_id][, path]); plume images are tagged with the RS domain id wherever
domain ids are used (routed knowledge distillation, per-domain validation).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

PLUME_INDEX = Path("data_index") / "industrial_plumes.csv"
# Order of the teacher logits and of the domain ids; must match the teachers dict of scripts/train_student.py.
DOMAIN_ORDER = ["cv", "rs", "uav"]
RS_DOMAIN_ID = DOMAIN_ORDER.index("rs")


class PlumeDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        dataset_root: Path,
        split: str,
        class_names: List[str],
        transform=None,
        domain_id: Optional[int] = None,
        return_path: bool = False,
    ) -> None:
        self.transform = transform
        self.domain_id = domain_id
        self.return_path = return_path
        self.samples: List[Tuple[Path, int]] = []
        self.sources: List[str] = []  # positive / negative folder of each sample
        label_to_id = {name: i for i, name in enumerate(class_names)}
        with Path(index_csv).open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"].lower() != split.lower():
                    continue
                if (row["label"] == "smoke") != (row["label_src"] == "positive"):
                    raise ValueError(f"{index_csv}: label {row['label']!r} disagrees with folder {row['label_src']!r}")
                self.samples.append((Path(dataset_root) / row["path"], label_to_id[row["label"]]))
                self.sources.append(row["label_src"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        item = (image, label)
        if self.domain_id is not None:
            item = item + (self.domain_id,)
        if self.return_path:
            item = item + (str(path),)
        return item
