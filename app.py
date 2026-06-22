# app.py
import os
from io import BytesIO
from datetime import datetime
from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image as PILImage
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from werkzeug.utils import secure_filename

from models import build_mobilenet_v2, build_leaf_detector
from tomato_labels import (
    TOMATO_CLASS_DESCRIPTIONS,
    TOMATO_CLASS_ORDER,
    TOMATO_QUICK_ID_GUIDE,
    normalize_tomato_label,
)
from typing import cast

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/mobilenet_v2_tomato_best.pth"
LEAF_MODEL_PATH = "leaf_detector.pth"
FEATURE_CLASSIFIER_PATH = "artifacts/tomato_classifier.pkl"
FEATURE_SCALER_PATH = "artifacts/tomato_feature_scaler.pkl"
FEATURE_IDX_TO_CLASS_PATH = "artifacts/tomato_idx_to_class.pkl"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

device = torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.60
LEAF_PROB_THRESHOLD = 0.70
LEAF_BLOCK_THRESHOLD = 0.35
LEAF_MODEL_LOADED = False
LEAF_MODEL_INFO = {}
MAX_BATCH_SIZE = 10
MAX_CONTENT_LENGTH_MB = 20
FEATURE_CLASSIFIER_LOADED = False
LEAF_IMG_SIZE = 160

# -------------------------------------------------------------
# Load Leaf Detector (REAL)
# -------------------------------------------------------------
LEAF_MODEL_PATH = "models/leaf_detector.pth"

leaf_model = build_leaf_detector(pretrained=True)
leaf_model = leaf_model.to(device)
leaf_model.eval()

if os.path.exists(LEAF_MODEL_PATH):
    leaf_checkpoint = torch.load(LEAF_MODEL_PATH, map_location=device)
    if isinstance(leaf_checkpoint, dict) and "model_state_dict" in leaf_checkpoint:
        leaf_model.load_state_dict(leaf_checkpoint["model_state_dict"])
        LEAF_PROB_THRESHOLD = float(leaf_checkpoint.get("leaf_threshold", LEAF_PROB_THRESHOLD))
        LEAF_IMG_SIZE = int(leaf_checkpoint.get("image_size", LEAF_IMG_SIZE))
        LEAF_MODEL_INFO = leaf_checkpoint.get("val_metrics", {})
    else:
        leaf_model.load_state_dict(leaf_checkpoint)
    LEAF_MODEL_LOADED = True
    print("Custom leaf detector loaded")
else:
    print("Leaf detector model not found - using ImageNet fallback")

# -------------------------------------------------------------
# Load Disease Models
# -------------------------------------------------------------
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
idx_to_class = ckpt["idx_to_class"]
num_classes = len(idx_to_class)
IMG_SIZE = int(ckpt.get("img_size", 224))

base_model = build_mobilenet_v2(num_classes=num_classes, pretrained=False)
base_model.load_state_dict(ckpt["model_state_dict"])
base_model = base_model.to(device)
base_model.eval()

feature_classifier = None
feature_scaler = None
feature_model = None
feature_extractor = None
feature_img_size = 224

if os.path.exists(FEATURE_CLASSIFIER_PATH) and os.path.exists(FEATURE_SCALER_PATH):
    feature_classifier = joblib.load(FEATURE_CLASSIFIER_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    if os.path.exists(FEATURE_IDX_TO_CLASS_PATH):
        idx_to_class = joblib.load(FEATURE_IDX_TO_CLASS_PATH)
        num_classes = len(idx_to_class)
    feature_model = build_mobilenet_v2(num_classes=num_classes, pretrained=True)
    feature_model = feature_model.to(device)
    feature_model.eval()
    from models.cnn_feature_extractor import CNNFeatureExtractor
    feature_extractor = CNNFeatureExtractor(feature_model).to(device)
    FEATURE_CLASSIFIER_LOADED = True
    IMG_SIZE = feature_img_size

# -------------------------------------------------------------
# Transforms
# -------------------------------------------------------------
leaf_transform = transforms.Compose([
    transforms.Resize((LEAF_IMG_SIZE, LEAF_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

disease_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_leaf_image(img: PILImage.Image) -> torch.Tensor:
    tensor = cast(torch.Tensor, leaf_transform(img))
    return tensor.unsqueeze(0)


def preprocess_disease_image(img: PILImage.Image) -> torch.Tensor:
    tensor = cast(torch.Tensor, disease_transform(img))
    return tensor.unsqueeze(0)


def save_upload(file_storage) -> tuple[str, str, str]:
    original_name = secure_filename(file_storage.filename or "upload.jpg")
    filename = f"{datetime.now().timestamp()}_{original_name}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(path)
    return original_name, filename, path


def _scores_from_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    if FEATURE_CLASSIFIER_LOADED and feature_classifier is not None and feature_scaler is not None and feature_extractor is not None:
        with torch.inference_mode():
            features = feature_extractor(img_tensor.to(device)).cpu().numpy()
        features_scaled = feature_scaler.transform(features)
        return feature_classifier.predict_proba(features_scaled)[0]

    with torch.inference_mode():
        cnn_logits = base_model(img_tensor.to(device))
        return torch.softmax(cnn_logits, dim=1).cpu().numpy()[0]


def read_cv2_image(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def classify_image(path: str, filename: str) -> dict:
    image_url = url_for("static", filename=f"uploads/{filename}")
    heatmap_url = None
    warning = None
    leaf_confidence = None
    leaf_status = "unknown"

    pil_image = PILImage.open(path).convert("RGB")
    leaf_tensor = preprocess_leaf_image(pil_image)
    disease_tensor = preprocess_disease_image(pil_image)

    with torch.inference_mode():
        leaf_logits = leaf_model(leaf_tensor)
        leaf_probs = torch.softmax(leaf_logits, dim=1)

    leaf_confidence = float(leaf_probs[0, 1])
    if LEAF_MODEL_LOADED and leaf_confidence < LEAF_BLOCK_THRESHOLD:
        prediction = "Not a leaf image"
        confidence = round(leaf_confidence * 100, 2)
        leaf_status = "blocked"
        warning = (
            f"This image does not look like a leaf to the detector "
            f"({leaf_confidence * 100:.1f}% leaf confidence). "
            "Upload a clearer tomato leaf image."
        )
        return {
            "original_name": filename.split("_", 1)[1] if "_" in filename else filename,
            "filename": filename,
            "image_url": image_url,
            "heatmap_url": None,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {},
            "warning": warning,
            "leaf_confidence": confidence,
            "is_leaf": False,
            "leaf_status": leaf_status,
        }

    if LEAF_MODEL_LOADED and leaf_confidence < LEAF_PROB_THRESHOLD:
        leaf_status = "warn"
        warning = (
            f"This image looks like a leaf but the detector is not fully sure "
            f"({leaf_confidence * 100:.1f}% leaf confidence). "
            "Disease prediction will still run."
        )
    elif LEAF_MODEL_LOADED:
        leaf_status = "pass"

    flipped_tensor = torch.flip(disease_tensor, dims=[3])
    original_scores = _scores_from_tensor(disease_tensor)
    flipped_scores = _scores_from_tensor(flipped_tensor)
    scores = (original_scores + flipped_scores) / 2.0
    pred_idx = int(np.argmax(scores))

    raw_confidence = float(scores[pred_idx])
    confidence = round(raw_confidence * 100, 2)

    probabilities = {
        normalize_tomato_label(idx_to_class[i]): round(float(scores[i]) * 100, 2)
        for i in range(len(scores))
    }
    probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    if raw_confidence < CONFIDENCE_THRESHOLD:
        pretty_prediction = normalize_tomato_label(idx_to_class[pred_idx])
        prediction = f"Best guess: {pretty_prediction}"
        if warning:
            warning += (
                f" Disease classification confidence is also low. "
                f"Best guess: {pretty_prediction} ({confidence}%)."
            )
        else:
            warning = (
                f"Low confidence. Best guess: {pretty_prediction} "
                f"({confidence}%). Try a clearer tomato leaf image."
            )
    else:
        prediction = normalize_tomato_label(idx_to_class[pred_idx])

        cam = gradcam.generate(disease_tensor, pred_idx)
        img = read_cv2_image(path)

        if img is not None:
            cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
            heatmap_input = np.clip(cam * 255, 0, 255).astype(np.uint8)
            cam = cv2.applyColorMap(heatmap_input, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, cam, 0.4, 0)

            heatmap_path = os.path.join(HEATMAP_FOLDER, filename)
            cv2.imwrite(heatmap_path, overlay)
            heatmap_url = url_for("static", filename=f"heatmaps/{filename}")

    return {
        "original_name": filename.split("_", 1)[1] if "_" in filename else filename,
        "filename": filename,
        "image_url": image_url,
        "heatmap_url": heatmap_url,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "warning": warning,
        "leaf_confidence": round(leaf_confidence * 100, 2),
        "is_leaf": True,
        "leaf_status": leaf_status,
    }

# -------------------------------------------------------------
# Grad-CAM
# -------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[:, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()

        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam /= cam.max() + 1e-8
        return cam

feature_layers = base_model.features
if not isinstance(feature_layers, nn.Sequential):
    raise TypeError("Expected base_model.features to be an nn.Sequential module.")
gradcam = GradCAM(base_model, feature_layers[-1])

# -------------------------------------------------------------
# Flask App
# -------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

CLASS_DESCRIPTIONS = {
    label: description for label, description in TOMATO_CLASS_DESCRIPTIONS.items()
}


def prettify_label(label: str) -> str:
    return normalize_tomato_label(label)


def is_allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


DISEASES = [
    {
        "name": normalize_tomato_label(class_name),
        "description": CLASS_DESCRIPTIONS.get(
            normalize_tomato_label(class_name),
            "Detected by the trained model.",
        ),
    }
    for _, class_name in sorted(idx_to_class.items())
]

QUICK_ID_GUIDE = TOMATO_QUICK_ID_GUIDE

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = image_url = heatmap_url = warning = None
    probabilities = {}
    results = []
    model_notice = None
    upload_error = None

    if num_classes != len(TOMATO_CLASS_ORDER):
        model_notice = (
            f"The current trained model in this app supports {num_classes} classes: "
            + ", ".join(normalize_tomato_label(name) for _, name in sorted(idx_to_class.items()))
            + "."
        )

    if request.method == "POST":
        uploaded_files = [
            file for file in request.files.getlist("images")
            if file and file.filename and is_allowed_file(file.filename)
        ]

        if not uploaded_files:
            single_file = request.files.get("image")
            if single_file and single_file.filename and is_allowed_file(single_file.filename):
                uploaded_files = [single_file]

        if request.files and not uploaded_files:
            upload_error = (
                "Please upload a valid image file (.jpg, .jpeg, .png, .bmp, or .webp)."
            )

        if len(uploaded_files) > MAX_BATCH_SIZE:
            warning = (
                f"You uploaded {len(uploaded_files)} images. "
                f"Only the first {MAX_BATCH_SIZE} were classified."
            )
            uploaded_files = uploaded_files[:MAX_BATCH_SIZE]

        for uploaded_file in uploaded_files:
            original_name, filename, path = save_upload(uploaded_file)
            result = classify_image(path, filename)
            result["original_name"] = original_name
            results.append(result)

        if results:
            first_result = results[0]
            prediction = first_result["prediction"]
            confidence = first_result["confidence"]
            image_url = first_result["image_url"]
            heatmap_url = first_result["heatmap_url"]
            probabilities = first_result["probabilities"]

            if first_result["warning"]:
                warning = (
                    f"{warning} {first_result['warning']}".strip()
                    if warning else first_result["warning"]
                )
            elif upload_error:
                warning = upload_error
        elif upload_error:
            warning = upload_error

    return render_template(
        "index.html",
        accuracy=98.6,
        classes=num_classes,
        images=1000,
        image_url=image_url,
        heatmap_url=heatmap_url,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        results=results,
        max_batch_size=MAX_BATCH_SIZE,
        diseases=DISEASES,
        quick_id_guide=QUICK_ID_GUIDE,
        model_notice=model_notice,
        warning=warning,
        upload_error=upload_error,
        year=datetime.now().year
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "leaf_detector_loaded": LEAF_MODEL_LOADED,
            "leaf_threshold": LEAF_PROB_THRESHOLD,
            "leaf_block_threshold": LEAF_BLOCK_THRESHOLD,
            "classes": num_classes,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if not file or not file.filename or not is_allowed_file(file.filename):
        return jsonify({"error": "Upload a valid image file."}), 400

    original_name, filename, path = save_upload(file)
    try:
        result = classify_image(path, filename)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(
        {
            "original_name": original_name,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "warning": result["warning"],
            "probabilities": result["probabilities"],
            "image_url": result["image_url"],
            "heatmap_url": result["heatmap_url"],
            "leaf_confidence": result.get("leaf_confidence"),
            "is_leaf": result.get("is_leaf"),
            "leaf_status": result.get("leaf_status"),
        }
    )

# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, use_reloader=False)
