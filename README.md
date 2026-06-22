# Tomato Disease Classification System

A machine learning system for classifying tomato diseases using CNN feature extraction and Fuzzy Min-Max Neural Network classification.

## Project Structure

```
├─ data/
│   └─ PlantVillage/
│       └─ Tomato/
│           ├─ Healthy/
│           ├─ Early_blight/
│           ├─ Late_blight/
│           ├─ Leaf_Mold/
│           ├─ Septoria_leaf_spot/
│           ├─ Bacterial_spot/
│           ├─ Tomato_mosaic_virus/
│           └─ Tomato_yellow_leaf_curl_virus/
├─ models/
│   ├─ __init__.py
│   ├─ cnn_feature_extractor.py
│   └─ fmn_classifier.py
├─ data_utils.py
├─ train_cnn.py
├─ train_fmn.py
├─ app.py
└─ README.md
```

## Features

- **CNN Feature Extraction**: Uses VGG16 pre-trained model for feature extraction
- **Fuzzy Min-Max Classification**: Custom FMN neural network for disease classification
- **Web Interface**: Flask-based web application for real-time predictions
- **Data Pipeline**: Utilities for loading, preprocessing, and augmenting image data

## Supported Classes

- Healthy
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Bacterial Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus

## Installation

### Requirements
- Python 3.11 recommended
- PyTorch and TorchVision for the current Flask app
- OpenCV
- Flask
- scikit-learn
- NumPy

### Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies for the current Flask app:
```bash
pip install -r requirements.txt
```

For the older TensorFlow-based training pipeline in `src/`, install:
```bash
pip install -r requirements-tensorflow.txt
```

3. Prepare dataset:
   - Download PlantVillage dataset
   - Organize images into `data/PlantVillage/Tomato/<class_name>/` directories

## Usage

### 1. Train CNN Feature Extractor

The scripts in `src/` use the legacy TensorFlow/Keras pipeline and require:

```bash
pip install -r requirements-tensorflow.txt
```

```bash
python train_cnn.py
```

This will:
- Load the tomato leaf dataset
- Run a fast MobileNetV2 fine-tune for the 10 tomato classes
- Save the best checkpoint to `checkpoints/mobilenet_v2_tomato_best.pth`

### 2. Train FMN Classifier

```bash
python train_fmn.py
```

This will:
- Extract features using trained CNN
- Train Fuzzy Min-Max Neural Network
- Save classifier to `models/fmn_classifier.pkl`

### 3. Run Web Application

The current `app.py` uses the PyTorch-based models and requires:

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

## Model Architecture

### CNN Feature Extractor
- Base: VGG16 (ImageNet pretrained weights)
- Input: 224×224×3 images
- Frozen base layers
- Custom extraction layers:
  - Flatten
  - Dense(512, relu) + Dropout(0.5)
  - Dense(256, relu) → 256-dim feature vector

### FMN Classifier
- Hyperbox-based pattern classification
- Fuzzy membership calculation
- Gamma parameter: 1.0
- Training epochs: 50

## API Endpoints

- `GET /` - Main upload page
- `POST /` - Submit image for prediction
- `GET /result` - Results page with prediction
- `GET /health` - Health check endpoint

## File Descriptions

### Models
- **cnn_feature_extractor.py**: CNN-based feature extraction
- **fmn_classifier.py**: Fuzzy Min-Max Neural Network implementation

### Training Scripts
- **train_cnn.py**: Train feature extractor
- **train_fmn.py**: Train classifier

### Utilities
- **data_utils.py**: Data loading and preprocessing functions
- **app.py**: Flask web application

## Configuration

Edit these variables in the training scripts:

```python
# train_cnn.py
EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# train_fmn.py
# (Automatically uses trained CNN model)
```

## Notes

- Images should be organized by disease class in subdirectories
- All images are resized to 224×224 pixels
- Pixel values are normalized to [0, 1] range
- Models are saved in the `models/` directory
- Uploaded images are stored in `static/uploads/`

## Future Improvements

- Add data augmentation in training pipeline
- Implement cross-validation
- Add confidence scores to predictions
- Create model evaluation dashboard
- Add batch prediction capability
- Implement model versioning

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation or training scripts.

## Usage

Upload tomato leaf images to the webapp to get disease predictions.
gunicorn==21.2.0
