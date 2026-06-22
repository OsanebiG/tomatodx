# data_utils.py
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from tomato_labels import normalize_tomato_label


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if CUDA not used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TomatoLeafDataset(Dataset):
    """
    Simple dataset for tomato leaf images.

    It takes a list of image paths, numerical labels, and a transform.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Always convert to RGB (handles grayscale if any)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def _load_split_samples(split_dir: Path) -> tuple[list[str], list[int], dict[int, str]]:
    """
    Load a single ImageFolder-style split directory and normalize class names.
    """
    dataset = datasets.ImageFolder(root=str(split_dir))
    idx_to_class = {
        idx: normalize_tomato_label(class_name)
        for class_name, idx in dataset.class_to_idx.items()
    }
    image_paths = [sample[0] for sample in dataset.samples]
    labels = [sample[1] for sample in dataset.samples]
    return image_paths, labels, idx_to_class


def _remap_labels(labels: list[int], source_idx_to_class: dict[int, str], target_class_to_idx: dict[str, int]) -> list[int]:
    return [target_class_to_idx[source_idx_to_class[label]] for label in labels]


def _limit_split_samples(
    paths: list[str],
    labels: list[int],
    max_samples_per_class: int | None,
    seed: int,
) -> tuple[list[str], list[int]]:
    if max_samples_per_class is None:
        return paths, labels

    rng = random.Random(seed)
    grouped: dict[int, list[str]] = {}
    for path, label in zip(paths, labels):
        grouped.setdefault(label, []).append(path)

    filtered_paths: list[str] = []
    filtered_labels: list[int] = []
    for label in sorted(grouped.keys()):
        samples = grouped[label]
        rng.shuffle(samples)
        selected = samples[:max_samples_per_class]
        filtered_paths.extend(selected)
        filtered_labels.extend([label] * len(selected))

    return filtered_paths, filtered_labels


def _split_small_dataset(
    image_paths: list[str],
    labels: list[int],
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """
    Split a tiny dataset by class while keeping at least one sample in train
    whenever possible.

    This avoids stratified split failures when each class only has a few images.
    """
    rng = random.Random(seed)
    grouped: dict[int, list[str]] = {}

    for path, label in zip(image_paths, labels):
        grouped.setdefault(label, []).append(path)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []

    for label in sorted(grouped.keys()):
        samples = grouped[label]
        rng.shuffle(samples)
        n = len(samples)

        if n <= 1:
            train_n, val_n, test_n = 1, 0, 0
        elif n == 2:
            train_n, val_n, test_n = 1, 0, 1
        elif n == 3:
            train_n, val_n, test_n = 1, 1, 1
        else:
            train_n = max(1, n - 2)
            val_n = 1
            test_n = n - train_n - val_n

        train_split = samples[:train_n]
        val_split = samples[train_n:train_n + val_n]
        test_split = samples[train_n + val_n:train_n + val_n + test_n]

        train_paths.extend(train_split)
        train_labels.extend([label] * len(train_split))
        val_paths.extend(val_split)
        val_labels.extend([label] * len(val_split))
        test_paths.extend(test_split)
        test_labels.extend([label] * len(test_split))

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def _merge_class_indices(*idx_maps: dict[int, str]) -> dict[int, str]:
    """
    Build a stable 0..N-1 index-to-class map from one or more split maps.
    """
    ordered_classes: list[str] = []
    for idx_map in idx_maps:
        for _, class_name in sorted(idx_map.items()):
            if class_name not in ordered_classes:
                ordered_classes.append(class_name)
    return {idx: class_name for idx, class_name in enumerate(ordered_classes)}


def get_data_splits(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    class_name_map: Dict[str, str] | None = None,
    max_samples_per_class: int | None = None,
    fast_mode: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Load the PlantVillage tomato subset and create 70/15/15 train/val/test splits.

    Returns:
        train_loader, val_loader, test_loader, idx_to_class
    """
    set_seed(seed)

    root_path = Path(data_dir)
    split_train = root_path / "train"
    split_valid = root_path / "valid"
    split_test = root_path / "test"
    has_explicit_splits = split_train.is_dir() and split_valid.is_dir()
    has_labeled_test = split_test.is_dir() and any(child.is_dir() for child in split_test.iterdir())

    if has_explicit_splits:
        train_paths, train_labels, train_idx_map = _load_split_samples(split_train)
        val_paths, val_labels, val_idx_map = _load_split_samples(split_valid)
        if has_labeled_test:
            test_paths, test_labels, test_idx_map = _load_split_samples(split_test)
        else:
            test_paths, test_labels, test_idx_map = [], [], {}

        idx_to_class = _merge_class_indices(train_idx_map, val_idx_map, test_idx_map)
        class_to_idx = {class_name: idx for idx, class_name in idx_to_class.items()}
        train_labels = _remap_labels(train_labels, train_idx_map, class_to_idx)
        val_labels = _remap_labels(val_labels, val_idx_map, class_to_idx)
        if test_paths:
            test_labels = _remap_labels(test_labels, test_idx_map, class_to_idx)

        train_paths, train_labels = _limit_split_samples(
            train_paths, train_labels, max_samples_per_class, seed
        )
        val_paths, val_labels = _limit_split_samples(
            val_paths, val_labels, max_samples_per_class, seed + 1
        )
        if test_paths:
            test_paths, test_labels = _limit_split_samples(
                test_paths, test_labels, max_samples_per_class, seed + 2
            )

        if not test_paths:
            test_paths = val_paths[:]
            test_labels = val_labels[:]
    else:
        base_dataset = datasets.ImageFolder(root=data_dir)
        class_to_idx = base_dataset.class_to_idx

        if class_name_map:
            allowed_classes = set(class_name_map.keys())
            selected_classes = sorted(
                [class_name for class_name in class_to_idx if class_name in allowed_classes]
            )
            remapped_class_to_idx = {
                class_name: idx for idx, class_name in enumerate(selected_classes)
            }
            idx_to_class = {
                idx: normalize_tomato_label(class_name_map[class_name])
                for class_name, idx in remapped_class_to_idx.items()
            }
            image_paths = []
            labels = []
            class_counts = {}
            max_per_class = 5  # limit per class for faster training
            for path, original_label in base_dataset.samples:
                original_class_name = base_dataset.classes[original_label]
                if original_class_name not in remapped_class_to_idx:
                    continue
                mapped_idx = remapped_class_to_idx[original_class_name]
                if class_counts.get(mapped_idx, 0) >= max_per_class:
                    continue
                image_paths.append(path)
                labels.append(mapped_idx)
                class_counts[mapped_idx] = class_counts.get(mapped_idx, 0) + 1
        else:
            idx_to_class = {
                v: normalize_tomato_label(k)
                for k, v in class_to_idx.items()
            }
            image_paths = [s[0] for s in base_dataset.samples]
            labels = [s[1] for s in base_dataset.samples]

            if max_samples_per_class is not None:
                filtered_paths = []
                filtered_labels = []
                class_counts = {}
                for path, label in zip(image_paths, labels):
                    if class_counts.get(label, 0) >= max_samples_per_class:
                        continue
                    filtered_paths.append(path)
                    filtered_labels.append(label)
                    class_counts[label] = class_counts.get(label, 0) + 1
                image_paths = filtered_paths
                labels = filtered_labels

        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = _split_small_dataset(
            image_paths,
            labels,
            seed=seed,
        )

    # Data augmentation for training; standard transforms for val/test
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if fast_mode:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    test_val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = TomatoLeafDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TomatoLeafDataset(val_paths, val_labels, transform=test_val_transform)
    test_dataset = TomatoLeafDataset(test_paths, test_labels, transform=test_val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, idx_to_class
