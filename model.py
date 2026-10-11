"""A CIFAR-10 CNN built from scratch, with unnormalized class logits."""

import torch
from torch import nn


class CNN(nn.Module):
    """Six convolutions and a configurable classifier for CIFAR images."""

    def __init__(self, num_classes=10, dropout=0.3, fc_hidden1=256,
                 fc_hidden2=None, activation='relu', channel_out1=32,
                 channel_out2=64, k_size=3, pooling_size=2, stride=1,
                 max_stride=2):
        super().__init__()
        activations = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                       'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        activation_layer = activations[activation]
        layers = []
        channels = 3
        for width in (channel_out1, channel_out2, channel_out2 * 2):
            for _ in range(2):
                layers.extend([
                    nn.Conv2d(channels, width, k_size, stride=stride, padding=k_size // 2, bias=False),
                    nn.BatchNorm2d(width),
                    activation_layer(),
                ])
                channels = width
            layers.extend([nn.MaxPool2d(pooling_size, stride=max_stride), nn.Dropout2d(dropout / 2)])
        self.features = nn.Sequential(*layers)
        self.features.eval()
        with torch.no_grad():
            feature_size = self.features(torch.zeros(1, 3, 32, 32)).numel()
        self.features.train()
        classifier = [nn.Flatten(), nn.Linear(feature_size, fc_hidden1),
                      activation_layer(), nn.Dropout(dropout)]
        if fc_hidden2 is not None:
            classifier.extend([nn.Linear(fc_hidden1, fc_hidden2),
                               activation_layer(), nn.Dropout(dropout)])
        classifier.append(nn.Linear(fc_hidden2 or fc_hidden1, num_classes))
        self.classifier = nn.Sequential(*classifier)
        self.apply(self.initialize)

    @staticmethod
    def initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_resnet(depth, num_classes=100):
    from torchvision import models

    models_by_depth = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}
    model = models_by_depth[depth](weights=None, num_classes=num_classes)
    # CIFAR images are 32x32, so preserve resolution in the first block.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    return model
