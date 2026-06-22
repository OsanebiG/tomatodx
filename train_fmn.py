# train_fmn.py
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from data_utils import get_data_splits, set_seed
from models.cnn_feature_extractor import build_mobilenet_v2, CNNFeatureExtractor
from models.fmn_classifier import FuzzyMinMaxClassifier, FMNConfig


def extract_features(
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run all images through CNN feature extractor and return numpy arrays.

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
    """
    feature_extractor.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = feature_extractor(images)  # (B, D)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx_to_class: Dict[int, str],
    title: str
) -> None:
    """
    Print accuracy, per-class precision/recall/F1, and confusion matrix.
    """
    print(f"\n=== {title} ===")
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[idx_to_class[i] for i in sorted(idx_to_class.keys())],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows: true, cols: predicted):")
    print(cm)


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
    batch_size = 64
    num_workers = 0
    seed = 42
    checkpoint_path = "checkpoints/mobilenet_v2_tomato_best.pth"

    # FMN hyperparameters (tune as needed)
    fmn_config = FMNConfig(
        max_hyperbox_size=0.3,
        gamma=1.0,
        n_epochs=1,
        shuffle=True,
        random_state=seed,
    )

    set_seed(seed)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_loader, val_loader, test_loader, idx_to_class = get_data_splits(
        data_dir=data_dir,
        img_size=224,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_samples_per_class=300,
    )
    num_classes = len(idx_to_class)

    # ---------------------------------------------------------
    # Load best CNN
    # ---------------------------------------------------------
    device = torch.device("cpu")

    base_model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.to(device)
    base_model.eval()

    # Build feature extractor from trained CNN
    feature_extractor = CNNFeatureExtractor(base_model).to(device)

    # ---------------------------------------------------------
    # Extract features for train/val/test
    # ---------------------------------------------------------
    print("Extracting CNN features for train/val/test...")
    X_train, y_train = extract_features(feature_extractor, train_loader, device)
    X_val, y_val = extract_features(feature_extractor, val_loader, device)
    X_test, y_test = extract_features(feature_extractor, test_loader, device)

    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape:   {X_val.shape}")
    print(f"Test features shape:  {X_test.shape}")

    # ---------------------------------------------------------
    # Scale features to [0,1] for FMN
    # ---------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Train FMN classifier on CNN features
    # ---------------------------------------------------------
    n_features = X_train_scaled.shape[1]
    fmn = FuzzyMinMaxClassifier(
        n_features=n_features,
        n_classes=num_classes,
        config=fmn_config,
    )

    print("Training Fuzzy Min–Max classifier on CNN features...")
    fmn.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # Evaluate baseline CNN (softmax) on test set
    # ---------------------------------------------------------
    print("\nEvaluating baseline CNN (softmax) on test set...")
    base_model.eval()
    all_logits = []
    all_test_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = base_model(images)  # (B, C)
            all_logits.append(logits.cpu())
            all_test_labels.append(labels)

    logits_test = torch.cat(all_logits, dim=0)
    y_test_cnn = logits_test.argmax(dim=1).numpy()
    y_test_np = torch.cat(all_test_labels, dim=0).numpy()

    evaluate_predictions(y_test_np, y_test_cnn, idx_to_class, title="Baseline CNN (softmax)")

    # ---------------------------------------------------------
    # Evaluate CNN–FMN hybrid on test set
    # ---------------------------------------------------------
    print("\nEvaluating CNN–FMN hybrid on test set...")
    y_test_fmn = fmn.predict(X_test_scaled)

    evaluate_predictions(y_test_np, y_test_fmn, idx_to_class, title="CNN–FMN Hybrid")

    # Optional: save FMN + scaler for deployment
    import joblib
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(fmn, "artifacts/tomato_classifier.pkl")
    joblib.dump(scaler, "artifacts/tomato_feature_scaler.pkl")
    joblib.dump(idx_to_class, "artifacts/tomato_idx_to_class.pkl")
    print("\nFMN model and scaler saved in 'artifacts/' directory.")


if __name__ == "__main__":
    main()
