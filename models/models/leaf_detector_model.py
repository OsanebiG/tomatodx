import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNetV2


def build_leaf_detector(pretrained: bool = True) -> MobileNetV2:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, 2),
    )
    return model
