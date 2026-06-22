# train_cnn.py
import os
from collections import Counter
from collections.abc import Sized
from typing import Protocol, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import MobileNetV2

from sklearn.metrics import accuracy_score

from data_utils import get_data_splits, set_seed
from models.cnn_feature_extractor import build_mobilenet_v2


class _DatasetWithLabels(Protocol):
    labels: list[int]


def _dataset_size(loader: DataLoader) -> int:
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Expected loader.dataset to implement __len__.")
    return len(dataset)


def _build_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [
        total / (num_classes * counts.get(class_idx, 1))
        for class_idx in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def _set_trainable_layers(model: MobileNetV2, train_backbone: bool) -> None:
    for param in model.parameters():
        param.requires_grad = False

    # Always train the classifier head.
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Light fine-tuning of the last MobileNetV2 blocks helps the minority classes
    # without turning the retrain into a long full-backbone job.
    if train_backbone:
        for block_idx, block in enumerate(model.features):
            if block_idx >= 14:
                for param in block.parameters():
                    param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)         # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    epoch_loss = running_loss / _dataset_size(loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    epoch_acc = float(accuracy_score(y_true, y_pred))

    return float(epoch_loss), epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / _dataset_size(loader)
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    epoch_acc = float(accuracy_score(y_true, y_pred))

    return {"loss": float(epoch_loss), "acc": epoch_acc}


def main():
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    data_dir = os.environ.get(
        "TOMATO_DATA_DIR",
        "data/PlantVillage/Tomato"
        if os.path.isdir("data/PlantVillage/Tomato")
        else os.path.expanduser("~/Downloads/Data leaf"),
    )
    batch_size = 128
    num_workers = 0
    img_size = 160
    phase_epochs = [4, 3]
    num_epochs = sum(phase_epochs)
    head_learning_rate = 2e-3
    finetune_learning_rate = 5e-5
    seed = 42
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_loader, val_loader, test_loader, idx_to_class = get_data_splits(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        class_name_map=None,
        max_samples_per_class=500,
        fast_mode=False,
    )
    num_classes = len(idx_to_class)
    train_dataset = cast(_DatasetWithLabels, train_loader.dataset)
    class_weights = _build_class_weights(train_dataset.labels, num_classes)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenet_v2(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ---------------------------------------------------------
    # Quick two-stage training:
    # 1) train only the classifier head
    # 2) lightly fine-tune the last backbone blocks
    # ---------------------------------------------------------
    best_val_acc = 0.0
    patience = 3
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(output_dir, "mobilenet_v2_tomato_best.pth")

    global_epoch = 0
    for phase_idx, phase_epoch_count in enumerate(phase_epochs):
        train_backbone = phase_idx > 0
        epochs_no_improve = 0
        _set_trainable_layers(model, train_backbone=train_backbone)

        learning_rate = finetune_learning_rate if train_backbone else head_learning_rate
        optimizer = optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )

        stage_name = "head" if not train_backbone else "fine-tune"
        print(f"\nStarting {stage_name} stage with lr={learning_rate:g}")

        for _ in range(phase_epoch_count):
            global_epoch += 1
            train_loss, train_acc = train_one_epoch(
                model, train_loader, device, criterion, optimizer
            )
            val_metrics = evaluate(model, val_loader, device, criterion)
            scheduler.step(val_metrics["acc"])

            print(
                f"Epoch {global_epoch}/{num_epochs} "
                f"- Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} "
                f"- Val loss: {val_metrics['loss']:.4f}, Val acc: {val_metrics['acc']:.4f}"
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "idx_to_class": idx_to_class,
                        "img_size": img_size,
                    },
                    best_ckpt_path,
                )
                print(f"  -> New best model saved to {best_ckpt_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        if epochs_no_improve >= patience:
            break

    print(f"Best val accuracy: {best_val_acc:.4f}")

    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, criterion)
        print(
            f"Test loss: {test_metrics['loss']:.4f}, "
            f"Test acc: {test_metrics['acc']:.4f}"
        )

    print("Training finished.")


if __name__ == "__main__":
    main()
