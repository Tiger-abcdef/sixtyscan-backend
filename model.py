# model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ParkinsonModel(nn.Module):
    """
    ResNet18 backbone with custom FC head:
    512 -> 128 -> ReLU -> Dropout(0.3) -> 2 classes
    """
    def __init__(self):
        super().__init__()

        # base model
        self.backbone = resnet18(weights=None)

        # replace final fully-connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
