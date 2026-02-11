"""
对比实验基线模型
使用 torchvision / segmentation_models_pytorch 提供的标准模型
支持: U-Net, DeepLabv3+, FCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """轻量级 U-Net (from scratch, 不依赖额外库)"""

    def __init__(self, num_classes=2, in_channels=3, base_channels=64):
        super().__init__()
        C = base_channels

        # Encoder
        self.enc1 = self._block(in_channels, C)
        self.enc2 = self._block(C, C * 2)
        self.enc3 = self._block(C * 2, C * 4)
        self.enc4 = self._block(C * 4, C * 8)

        # Bottleneck
        self.bottleneck = self._block(C * 8, C * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(C * 16, C * 8, 2, stride=2)
        self.dec4 = self._block(C * 16, C * 8)
        self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec3 = self._block(C * 8, C * 4)
        self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec2 = self._block(C * 4, C * 2)
        self.up1 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec1 = self._block(C * 2, C)

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(C, num_classes, 1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def get_trainable_params(self):
        return [{"params": p, "name": n} for n, p in self.named_parameters()]


class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with ResNet backbone (使用 torchvision)"""

    def __init__(self, num_classes=2, backbone="resnet50", pretrained=True):
        super().__init__()
        import torchvision.models.segmentation as seg

        if backbone == "resnet50":
            self.model = seg.deeplabv3_resnet50(
                weights="DEFAULT" if pretrained else None,
                num_classes=num_classes,
            )
        elif backbone == "resnet101":
            self.model = seg.deeplabv3_resnet101(
                weights="DEFAULT" if pretrained else None,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        out = self.model(x)
        return out["out"]

    def get_trainable_params(self):
        return [{"params": p, "name": n} for n, p in self.named_parameters()]


class FCN(nn.Module):
    """FCN with ResNet backbone (使用 torchvision)"""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        import torchvision.models.segmentation as seg

        self.model = seg.fcn_resnet50(
            weights="DEFAULT" if pretrained else None,
            num_classes=num_classes,
        )

    def forward(self, x):
        out = self.model(x)
        return out["out"]

    def get_trainable_params(self):
        return [{"params": p, "name": n} for n, p in self.named_parameters()]


def build_baseline(model_name, num_classes=2, **kwargs):
    """
    构建基线模型

    Args:
        model_name: "unet", "deeplabv3plus", "fcn"
        num_classes: 类别数
    """
    model_name = model_name.lower().replace("-", "").replace("_", "")
    if model_name == "unet":
        return UNet(num_classes=num_classes, **kwargs)
    elif model_name in ("deeplabv3plus", "deeplabv3+", "deeplabv3"):
        return DeepLabV3Plus(num_classes=num_classes, **kwargs)
    elif model_name == "fcn":
        return FCN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown baseline: {model_name}. Choose from: unet, deeplabv3plus, fcn")
