from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from models.leaf_detector_model import build_leaf_detector


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 128
BATCH_SIZE = 96
EPOCHS = 5
SEED = 42
MAX_LEAF_SAMPLES = None
BEST_THRESHOLD_STEPS = 101

LEAF_ROOT = Path("data/PlantVillage/Tomato/train")
NON_LEAF_ROOT = Path("data/leaf_detector/non_leaf")
OUTPUT_PATH = Path("models/leaf_detector.pth")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_images(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return files


class LeafBinaryDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_loaders(seed: int = 42) -> tuple[DataLoader, DataLoader]:
    leaf_paths = collect_images(LEAF_ROOT)
    non_leaf_paths = collect_images(NON_LEAF_ROOT)

    if not leaf_paths:
        raise FileNotFoundError(f"No leaf images found in {LEAF_ROOT.resolve()}")
    if not non_leaf_paths:
        raise FileNotFoundError(f"No non-leaf images found in {NON_LEAF_ROOT.resolve()}")

    rng = random.Random(seed)
    if MAX_LEAF_SAMPLES is not None and len(leaf_paths) > MAX_LEAF_SAMPLES:
        leaf_paths = rng.sample(leaf_paths, MAX_LEAF_SAMPLES)

    image_paths = leaf_paths + non_leaf_paths
    labels = [1] * len(leaf_paths) + [0] * len(non_leaf_paths)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = LeafBinaryDataset(train_paths, train_labels, transform=train_transform)
    val_ds = LeafBinaryDataset(val_paths, val_labels, transform=val_transform)

    class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    cpu_count = os.cpu_count() or 1
    num_workers = 0 if cpu_count <= 2 else min(4, cpu_count // 2)
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probs)
    y_pred = (y_score >= 0.5).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = float((y_pred == y_true).mean())

    return (
        total_loss / max(total, 1),
        acc,
        float(precision),
        float(recall),
        float(f1),
        float(bal_acc),
        y_true,
        y_score,
    )


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float, float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_bal_acc = 0.0
    best_acc = 0.0

    for threshold in np.linspace(0.05, 0.95, BEST_THRESHOLD_STEPS):
        y_pred = (y_score >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = float((y_pred == y_true).mean())

        if f1 > best_f1 or (np.isclose(f1, best_f1) and bal_acc > best_bal_acc):
            best_threshold = float(threshold)
            best_f1 = float(f1)
            best_precision = float(precision)
            best_recall = float(recall)
            best_bal_acc = float(bal_acc)
            best_acc = acc

    return best_threshold, best_precision, best_recall, best_f1, best_bal_acc, best_acc


def main() -> None:
    set_seed(SEED)
    train_loader, val_loader = build_loaders(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_leaf_detector(pretrained=True).to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[14:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features[14:].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_val_f1 = 0.0
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_bal_acc, y_true, y_score = evaluate(
            model, val_loader, device, criterion
        )
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} - "
            f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, "
            f"Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, "
            f"Val f1: {val_f1:.4f}, Val bal acc: {val_bal_acc:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"  -> Saved best leaf detector to {OUTPUT_PATH}")

    if OUTPUT_PATH.exists():
        best_state = torch.load(OUTPUT_PATH, map_location=device)
        model.load_state_dict(best_state)
        _, _, _, _, _, _, y_true, y_score = evaluate(model, val_loader, device, criterion)
        best_threshold, thr_precision, thr_recall, thr_f1, thr_bal_acc, thr_acc = find_best_threshold(
            y_true, y_score
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "leaf_threshold": best_threshold,
            "image_size": IMAGE_SIZE,
            "class_names": {0: "non_leaf", 1: "leaf"},
            "val_metrics": {
                "accuracy": thr_acc,
                "precision": thr_precision,
                "recall": thr_recall,
                "f1": thr_f1,
                "balanced_accuracy": thr_bal_acc,
            },
        }
        torch.save(checkpoint, OUTPUT_PATH)
        print(
            f"Best validation F1: {best_val_f1:.4f} | "
            f"Chosen leaf threshold: {best_threshold:.2f}"
        )
        print(
            f"Threshold metrics -> acc: {thr_acc:.4f}, precision: {thr_precision:.4f}, "
            f"recall: {thr_recall:.4f}, f1: {thr_f1:.4f}, bal acc: {thr_bal_acc:.4f}"
        )
    else:
        print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
