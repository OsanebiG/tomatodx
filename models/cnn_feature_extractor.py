import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNetV2


def build_mobilenet_v2(num_classes: int, pretrained: bool = True) -> MobileNetV2:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    return model


class CNNFeatureExtractor(nn.Module):
    def __init__(self, base_model: MobileNetV2) -> None:
        super().__init__()
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
