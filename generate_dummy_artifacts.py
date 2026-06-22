import os
import torch
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.cnn_feature_extractor import build_mobilenet_v2
from models.fmn_classifier import FuzzyMinMaxClassifier, FMNConfig
from tomato_labels import TOMATO_CLASS_DISPLAY_NAMES, TOMATO_CLASS_ORDER

# --------------------------------------------------
# Paths
# --------------------------------------------------
CHECKPOINT_PATH = "checkpoints/mobilenet_v2_tomato_best.pth"
FMN_PATH = "artifacts/fmn_model.pkl"
SCALER_PATH = "artifacts/feature_scaler.pkl"
IDX_TO_CLASS_PATH = "artifacts/idx_to_class.pkl"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

device = torch.device("cpu")

# --------------------------------------------------
# Fake class labels
# --------------------------------------------------
idx_to_class = {
    idx: TOMATO_CLASS_DISPLAY_NAMES[class_name]
    for idx, class_name in enumerate(TOMATO_CLASS_ORDER)
}

num_classes = len(idx_to_class)

# --------------------------------------------------
# Create CNN checkpoint
# --------------------------------------------------
print("Creating dummy CNN checkpoint...")

model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
model.to(device)
model.eval()

checkpoint = {
    "model_state_dict": model.state_dict(),
    "idx_to_class": idx_to_class
}

torch.save(checkpoint, CHECKPOINT_PATH)

# --------------------------------------------------
# Create fake feature scaler
# --------------------------------------------------
print("Creating dummy scaler...")

scaler = MinMaxScaler()
dummy_features = np.random.rand(100, 1280)  # MobileNetV2 feature size
scaler.fit(dummy_features)

joblib.dump(scaler, SCALER_PATH)

# --------------------------------------------------
# Create fake FMN classifier
# --------------------------------------------------
print("Creating dummy FMN classifier...")

fmn = FuzzyMinMaxClassifier(
    n_features=1280,
    n_classes=num_classes,
    config=FMNConfig(n_epochs=1)
)

X_dummy = scaler.transform(dummy_features)
y_dummy = np.random.randint(0, num_classes, size=100)
fmn.fit(X_dummy, y_dummy)

joblib.dump(fmn, FMN_PATH)

# --------------------------------------------------
# Save idx_to_class separately
# --------------------------------------------------
joblib.dump(idx_to_class, IDX_TO_CLASS_PATH)

print("✅ Dummy artifacts created successfully!")
