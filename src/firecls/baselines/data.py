"""Domain-wise datasets built from the same CSV indices the teachers and student used."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from firecls.baselines.protocol import CLASSES, DOMAINS
from firecls.config import get_dataset_specs, get_repo_root
from firecls.data.dataset import ImageClassificationCSVDataset


def index_csv_path(domain: str, root: Path | None = None) -> Path:
    specs = get_dataset_specs(root)
    return (root or get_repo_root()) / "data_index" / f"{specs[domain].name.lower()}_tdml.csv"


def build_domain_datasets(
    split: str,
    transform,
    dataset_root: Path | None = None,
    domains: list[str] | None = None,
    return_path: bool = False,
    index_root: Path | None = None,
) -> dict[str, ImageClassificationCSVDataset]:
    """One dataset per domain. ``dataset_root`` is the parent of ``datasets/``.

    Image paths in the committed CSVs were written on Windows; they are re-based beneath each
    domain's ``images_root`` so the same index works on Linux, WSL, containers and Jetson.
    """
    specs = get_dataset_specs(dataset_root)
    datasets = {}
    for domain in domains or DOMAINS:
        csv_path = index_csv_path(domain, index_root)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} missing; run scripts/prepare_tdml_classification.py first.")
        dataset = ImageClassificationCSVDataset(
            csv_path,
            split,
            CLASSES,
            transform=transform,
            return_path=return_path,
            images_root=specs[domain].images_root,
        )
        if len(dataset) == 0:
            raise ValueError(f"No '{split}' samples for domain '{domain}' in {csv_path}")
        datasets[domain] = dataset
    return datasets


def combined_train_loader(
    transform, batch_size: int, num_workers: int, dataset_root: Path | None = None, seed_generator=None
) -> DataLoader:
    """All three domains' training splits concatenated, exactly as the student was trained."""
    datasets = build_domain_datasets("train", transform, dataset_root)
    return DataLoader(
        ConcatDataset(list(datasets.values())),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=seed_generator,
    )


def domain_loaders(
    split: str,
    transform,
    batch_size: int,
    num_workers: int,
    dataset_root: Path | None = None,
    return_path: bool = False,
) -> dict[str, DataLoader]:
    datasets = build_domain_datasets(split, transform, dataset_root, return_path=return_path)
    return {
        domain: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        for domain, dataset in datasets.items()
    }
