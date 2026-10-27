import torch.nn as nn
from torchvision import models


def build_orange_classifier(num_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model
