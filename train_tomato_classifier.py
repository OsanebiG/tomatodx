import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_utils import get_data_splits, set_seed
from models.cnn_feature_extractor import build_mobilenet_v2, CNNFeatureExtractor


@dataclass
class ClassifierBundle:
    classifier: LogisticRegression
    scaler: StandardScaler
    idx_to_class: dict[int, str]
    img_size: int


def extract_features(
    feature_extractor: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    feature_extractor.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = feature_extractor(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main() -> None:
    data_dir = os.environ.get(
        "TOMATO_DATA_DIR",
        "data/PlantVillage/Tomato"
        if os.path.isdir("data/PlantVillage/Tomato")
        else os.path.expanduser("~/Downloads/Data leaf"),
    )
    seed = 42
    batch_size = 128
    num_workers = 0
    img_size = 224
    max_samples_per_class = 200
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    set_seed(seed)

    train_loader, val_loader, test_loader, idx_to_class = get_data_splits(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_samples_per_class=max_samples_per_class,
        fast_mode=True,
    )

    num_classes = len(idx_to_class)
    device = torch.device("cpu")

    backbone = build_mobilenet_v2(num_classes=num_classes, pretrained=True)
    feature_extractor = CNNFeatureExtractor(backbone).to(device)

    print("Extracting train features...")
    X_train, y_train = extract_features(feature_extractor, train_loader, device)
    print("Extracting validation features...")
    X_val, y_val = extract_features(feature_extractor, val_loader, device)
    print("Extracting test features...")
    X_test, y_test = extract_features(feature_extractor, test_loader, device)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
    )

    print("Training logistic regression classifier...")
    classifier.fit(X_train_scaled, y_train)

    val_pred = classifier.predict(X_val_scaled)
    test_pred = classifier.predict(X_test_scaled)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nClassification report (test):")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[idx_to_class[i] for i in sorted(idx_to_class.keys())],
            zero_division=0,
        )
    )
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, test_pred))

    bundle = ClassifierBundle(
        classifier=classifier,
        scaler=scaler,
        idx_to_class=idx_to_class,
        img_size=img_size,
    )

    joblib.dump(bundle.classifier, output_dir / "tomato_classifier.pkl")
    joblib.dump(bundle.scaler, output_dir / "tomato_feature_scaler.pkl")
    joblib.dump(bundle.idx_to_class, output_dir / "tomato_idx_to_class.pkl")
    print(f"\nSaved classifier artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
