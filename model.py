
"""
ResNet50 for binary classification (OCT images)
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_resnet50_binary(pretrained: bool = True) -> nn.Module:
    """
    ResNet50 with Dropout and final FC layer adapted for 2 classes (binary classification).
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    return model