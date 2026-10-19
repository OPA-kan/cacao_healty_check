"""ResNet50 variants used in the cacao disease comparison."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


class ChannelAttention(nn.Module):
    """Channel attention: learns which feature channels matter."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=(2, 3))
        max_value = torch.amax(x, dim=(2, 3))
        weights = torch.sigmoid(self.mlp(avg) + self.mlp(max_value))
        return weights[:, :, None, None]


class SpatialAttention(nn.Module):
    """Spatial attention: learns where useful evidence is located."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, max_value], dim=1)))


class CBAM(nn.Module):
    """The standard sequential CBAM module."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        x_channel = self.channel(x) * x
        return self.spatial(x_channel) * x_channel


class LGFCBAM(nn.Module):
    """CBAM with a learnable gate for channel/spatial feature fusion.

    The two attention paths start from the same feature map. A small MLP
    predicts two normalized weights for each sample, then combines the
    channel-attended and spatial-attended maps.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()
        gate_hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        channel_map = self.channel(x)
        spatial_map = self.spatial(x)
        channel_features = channel_map * x
        spatial_features = spatial_map * x
        descriptors = torch.cat(
            [torch.mean(channel_features, dim=(2, 3)),
             torch.mean(spatial_features, dim=(2, 3))],
            dim=1,
        )
        weights = torch.softmax(self.gate(descriptors), dim=1)
        return (
            weights[:, 0, None, None, None] * channel_features
            + weights[:, 1, None, None, None] * spatial_features
        )


ModelName = Literal["resnet50", "cbam", "lgf_cbam"]


class ResNet50AttentionClassifier(nn.Module):
    """ResNet50 classifier with optional attention after layer4."""

    def __init__(
        self,
        num_classes: int,
        variant: ModelName = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.attention: nn.Module
        if variant == "cbam":
            self.attention = CBAM(2048)
        elif variant == "lgf_cbam":
            self.attention = LGFCBAM(2048)
        else:
            self.attention = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)
        self.variant = variant

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(
    variant: ModelName,
    num_classes: int,
    pretrained: bool = True,
) -> ResNet50AttentionClassifier:
    """Build one of the three comparison models."""

    return ResNet50AttentionClassifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
    )

