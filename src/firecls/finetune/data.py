"""Training and validation data for the plume fine-tuning.

Training data: FASDD train of every domain the model was originally trained on (RS only for the RS teacher; CV + RS + UAV for
the students and baselines) plus, unless `include_plumes` is False (control runs), the plume train split, which is tagged as
the RS domain. Validation: FASDD CV and UAV val (4-class accuracy), FASDD-RS val and plume val (binary).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from firecls.baselines.data import index_csv_path
from firecls.config import get_dataset_specs
from firecls.data.dataset import ImageClassificationCSVDataset
from firecls.data.plumes import DOMAIN_ORDER, PLUME_INDEX, RS_DOMAIN_ID, PlumeDataset


class SourceBalancedBatchSampler:
    """Pre-registered fallback for the RS teacher: every batch holds the same number of images from each source.

    `sources` are lists of indices into the training ConcatDataset (e.g. the FASDD-RS part and the plume part). Each batch takes
    `batch_size / len(sources)` images from every source, drawn without replacement from a shuffled queue that is reshuffled when
    it runs out (the smaller source is therefore passed more than once per epoch). One epoch has `num_batches` batches, so it draws
    about as many images as the unbalanced epoch would.
    """

    def __init__(self, sources: List[List[int]], batch_size: int, num_batches: int, seed: int) -> None:
        if batch_size % len(sources):
            raise ValueError(f"batch size {batch_size} is not divisible by the {len(sources)} sources")
        self.sources = [list(s) for s in sources]
        self.per_source = batch_size // len(sources)
        self.num_batches = num_batches
        self.rng = torch.Generator().manual_seed(seed)
        self.queues: List[List[int]] = [[] for _ in sources]

    def _take(self, k: int) -> List[int]:
        queue = self.queues[k]
        while len(queue) < self.per_source:
            order = torch.randperm(len(self.sources[k]), generator=self.rng).tolist()
            queue.extend(self.sources[k][i] for i in order)
        taken, self.queues[k] = queue[: self.per_source], queue[self.per_source:]
        return taken

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = [i for k in range(len(self.sources)) for i in self._take(k)]
            order = torch.randperm(len(batch), generator=self.rng).tolist()  # mix the sources inside the batch
            yield [batch[i] for i in order]


def source_balanced_batch_sampler(dataset: ConcatDataset, part_names: List[str], groups: List[List[str]],
                                  batch_size: int, seed: int) -> SourceBalancedBatchSampler:
    """Sources = groups of named parts of `dataset` (as returned by build_train_dataset), e.g. [["fasdd_rs"], ["plume"]]."""
    offsets, start = {}, 0
    for name, part in zip(part_names, dataset.datasets):
        offsets[name] = range(start, start + len(part))
        start += len(part)
    sources = [[i for name in group for i in offsets[name]] for group in groups]
    return SourceBalancedBatchSampler(sources, batch_size, len(dataset) // batch_size, seed)


def fasdd_domain(key: str, split: str, classes: List[str], transform, dataset_root: Path, index_root: Path,
                 domain_id: Optional[int] = None) -> ImageClassificationCSVDataset:
    spec = get_dataset_specs(dataset_root)[key]
    return ImageClassificationCSVDataset(
        index_csv_path(key, index_root), split, classes, transform=transform, images_root=spec.images_root,
        domain_id=domain_id,
    )


def plume_domain(split: str, classes: List[str], transform, dataset_root: Path, index_root: Path,
                 domain_id: Optional[int] = None) -> PlumeDataset:
    return PlumeDataset(index_root / PLUME_INDEX, dataset_root, split, classes, transform=transform, domain_id=domain_id)


RS_PART_NAMES = ("fasdd_rs", "plume")


def build_train_dataset(domains: List[str], classes: List[str], transform, dataset_root: Path, index_root: Path,
                        include_plumes: bool, tag_domains: bool) -> tuple[ConcatDataset, Dict[str, int]]:
    """Concatenation of cv / rs / uav train sets (in DOMAIN_ORDER) with the plume train set added to the RS domain.

    With `tag_domains`, every sample also yields its domain id; plume samples get RS_DOMAIN_ID, so routed distillation takes
    their soft target from the RS teacher.
    """
    parts: List[Dataset] = []
    sizes: Dict[str, int] = {}
    for key in [d for d in DOMAIN_ORDER if d in domains]:
        domain_id = DOMAIN_ORDER.index(key) if tag_domains else None
        dataset = fasdd_domain(key, "train", classes, transform, dataset_root, index_root, domain_id)
        parts.append(dataset)
        sizes[f"fasdd_{key}"] = len(dataset)
        if key == "rs" and include_plumes:
            plumes = plume_domain("train", classes, transform, dataset_root, index_root,
                                  RS_DOMAIN_ID if tag_domains else None)
            parts.append(plumes)
            sizes["plume"] = len(plumes)
    return ConcatDataset(parts), sizes


def build_val_datasets(domains: List[str], classes: List[str], transform, dataset_root: Path,
                       index_root: Path) -> Dict[str, Dataset]:
    """cv / uav (if the model has them), FASDD-RS val and plume val. Plume val is always included, also for the control."""
    sets: Dict[str, Dataset] = {}
    for key in ("cv", "uav"):
        if key in domains:
            sets[key] = fasdd_domain(key, "val", classes, transform, dataset_root, index_root)
    sets["rs_fasdd"] = fasdd_domain("rs", "val", classes, transform, dataset_root, index_root)
    sets["rs_plume"] = plume_domain("val", classes, transform, dataset_root, index_root)
    return sets


def rs_class_balanced_sampler(dataset: ConcatDataset, part_names: List[str], seed: int) -> WeightedRandomSampler:
    """Pre-registered fallback: class-balanced sampling of the RS portion only (FASDD-RS train + plume train).

    `part_names` names the parts of `dataset` in order (as returned by `build_train_dataset`); `fasdd_rs` and `plume` are the RS
    portion. Inside the RS portion every four-class label gets the same total weight; the RS portion keeps the total weight it
    has under normal sampling (its number of images), and every CV / UAV image keeps weight 1. One epoch draws len(dataset)
    samples.
    """
    labels, is_rs = [], []
    for name, part in zip(part_names, dataset.datasets):
        part_labels = [label for _, label in part.samples]
        labels += part_labels
        is_rs += [name in RS_PART_NAMES] * len(part_labels)
    counts = Counter(l for l, r in zip(labels, is_rs) if r)
    rs_total = sum(counts.values())
    weights = []
    for label, rs in zip(labels, is_rs):
        weights.append(rs_total / (len(counts) * counts[label]) if rs else 1.0)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
