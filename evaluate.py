# evaluate.py
# Run this script from your project root directory.
# It will print all the metrics you need for Chapter Four.
#
# Usage:
#   python evaluate.py
#   python evaluate.py --data_dir "C:/path/to/your/Data leaf"
#
# Output is also saved to: evaluation_results.txt

import os
import sys
import argparse
import datetime

import numpy as np
import torch
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from data_utils import get_data_splits, set_seed
from models.cnn_feature_extractor import build_mobilenet_v2, CNNFeatureExtractor


# ─────────────────────────────────────────────
# Config — edit these paths if needed
# ─────────────────────────────────────────────
DEFAULT_DATA_DIR = (
    "data/PlantVillage/Tomato"
    if os.path.isdir("data/PlantVillage/Tomato")
    else os.path.expanduser("~/Downloads/Data leaf")
)
CNN_CHECKPOINT   = "checkpoints/mobilenet_v2_tomato_best.pth"
FMN_PKL          = "artifacts/tomato_classifier.pkl"
SCALER_PKL       = "artifacts/tomato_feature_scaler.pkl"
IDX_TO_CLASS_PKL = "artifacts/tomato_idx_to_class.pkl"
OUTPUT_TXT       = "evaluation_results.txt"

SEED       = 42
BATCH_SIZE = 64
IMG_SIZE   = 160      # must match what was used in train_cnn.py
MAX_PER_CLASS = 500   # must match train_cnn.py setting


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def section(title: str) -> str:
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}"


def fmt_cm(cm: np.ndarray, class_names: list[str]) -> str:
    """Pretty-print confusion matrix with row/col labels."""
    col_w = max(max(len(n) for n in class_names), 6)
    header = " " * (col_w + 2) + "  ".join(n[:col_w].rjust(col_w) for n in class_names)
    lines = [header]
    for i, row in enumerate(cm):
        row_str = class_names[i][:col_w].ljust(col_w) + "  " + \
                  "  ".join(str(v).rjust(col_w) for v in row)
        lines.append(row_str)
    return "\n".join(lines)


def evaluate_model(y_true, y_pred, class_names, title):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    lines = [section(title)]
    lines.append(f"\nOverall Accuracy : {acc * 100:.2f}%")
    lines.append(f"Weighted Precision: {w_prec * 100:.2f}%")
    lines.append(f"Weighted Recall   : {w_rec * 100:.2f}%")
    lines.append(f"Weighted F1-Score : {w_f1 * 100:.2f}%")

    lines.append("\n--- Per-Class Metrics ---")
    lines.append(f"{'Class':<42} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 85)
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:<42} {prec[i]*100:>9.2f}% {rec[i]*100:>9.2f}% {f1[i]*100:>9.2f}% {int(sup[i]):>10}"
        )

    lines.append("\n--- Confusion Matrix (rows=True, cols=Predicted) ---")
    lines.append(fmt_cm(cm, class_names))

    lines.append("\n--- sklearn classification_report ---")
    lines.append(report)

    return "\n".join(lines), acc, cm


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN and CNN-FMN on test set")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Path to PlantVillage tomato data folder")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data directory : {data_dir}")
    print(f"CNN checkpoint : {CNN_CHECKPOINT}")
    print(f"FMN artifact   : {FMN_PKL}")

    if not os.path.isdir(data_dir):
        sys.exit(f"\n[ERROR] Data directory not found: {data_dir}\n"
                 f"Pass the correct path with --data_dir")

    if not os.path.exists(CNN_CHECKPOINT):
        sys.exit(f"\n[ERROR] CNN checkpoint not found: {CNN_CHECKPOINT}\n"
                 f"Run train_cnn.py first.")

    set_seed(SEED)

    # ── Data ──────────────────────────────────
    print("\nLoading data splits...")
    _, _, test_loader, idx_to_class = get_data_splits(
        data_dir=data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=SEED,
        max_samples_per_class=MAX_PER_CLASS,
        fast_mode=False,
    )
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Test samples: {len(test_loader.dataset)}")

    device = torch.device("cpu")

    # ── Load CNN ──────────────────────────────
    print("\nLoading CNN model...")
    ckpt = torch.load(CNN_CHECKPOINT, map_location=device)
    model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── CNN Predictions ───────────────────────
    print("Running CNN inference on test set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred_cnn = np.concatenate(all_preds)

    cnn_text, cnn_acc, cnn_cm = evaluate_model(y_true, y_pred_cnn, class_names, "CNN-Only Baseline Results")

    # ── CNN-FMN Predictions ───────────────────
    fmn_text = ""
    if os.path.exists(FMN_PKL) and os.path.exists(SCALER_PKL):
        print("Loading FMN classifier...")
        fmn = joblib.load(FMN_PKL)
        scaler = joblib.load(SCALER_PKL)

        # Re-extract features using the same CNN backbone
        feature_extractor = CNNFeatureExtractor(model).to(device)
        feature_extractor.eval()

        print("Extracting CNN features for FMN...")
        all_feats, all_fmn_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                feats = feature_extractor(images.to(device)).cpu().numpy()
                all_feats.append(feats)
                all_fmn_labels.append(labels.numpy())

        X_test = np.concatenate(all_feats, axis=0)
        y_fmn_true = np.concatenate(all_fmn_labels, axis=0)
        X_test_scaled = scaler.transform(X_test)

        print("Running FMN predictions...")
        y_pred_fmn = fmn.predict(X_test_scaled)

        fmn_text, fmn_acc, fmn_cm = evaluate_model(
            y_fmn_true, y_pred_fmn, class_names, "CNN-FMN Hybrid Results"
        )

        gain_text = (
            f"\n{'=' * 60}\n  IMPROVEMENT: CNN-FMN vs CNN-Only\n{'=' * 60}\n"
            f"CNN-Only Accuracy : {cnn_acc * 100:.2f}%\n"
            f"CNN-FMN Accuracy  : {fmn_acc * 100:.2f}%\n"
            f"Improvement       : {(fmn_acc - cnn_acc) * 100:+.2f} percentage points\n"
        )
    else:
        fmn_text = "\n[SKIP] FMN artifacts not found — run train_fmn.py or train_tomato_classifier.py first."
        gain_text = ""

    # ── Write output ──────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_output = (
        f"Tomato Disease Classification — Evaluation Report\n"
        f"Generated: {timestamp}\n"
        f"Data dir : {data_dir}\n"
        + cnn_text
        + "\n"
        + fmn_text
        + "\n"
        + gain_text
    )

    print(full_output)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\n[DONE] Results saved to: {os.path.abspath(OUTPUT_TXT)}")
    print("Paste the contents of that file back to Claude to complete Chapter Four.")


if __name__ == "__main__":
    main()
