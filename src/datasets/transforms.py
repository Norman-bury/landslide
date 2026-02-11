"""
数据增强流水线
支持训练时的多种增强策略，降低模型对单一成像风格的依赖

DINOv3 归一化参数:
- Web预训练 (LVD-1689M): mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)  # ImageNet标准
- 卫星预训练 (SAT-493M): mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# 预定义归一化参数
NORM_IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
NORM_SAT493M = {"mean": [0.430, 0.411, 0.296], "std": [0.213, 0.156, 0.143]}


def get_train_transforms(image_size=512, norm_type="imagenet"):
    """
    训练集增强: 色彩/对比度变化、噪声/模糊、翻转/旋转等
    
    Args:
        image_size: 输出图像尺寸
        norm_type: "imagenet" (web预训练) 或 "sat" (卫星预训练)
    """
    norm = NORM_SAT493M if norm_type == "sat" else NORM_IMAGENET
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # 色彩和对比度变化
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5
        ),
        # 噪声与模糊
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
        ], p=0.3),
        # 分辨率变化 (模拟不同传感器分辨率)
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.9), p=1.0),
        ], p=0.2),
        # 阴影模拟
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        # 归一化 + 转tensor
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=512, norm_type="imagenet"):
    """
    验证/测试集: 仅resize和归一化
    
    Args:
        image_size: 输出图像尺寸
        norm_type: "imagenet" (web预训练) 或 "sat" (卫星预训练)
    """
    norm = NORM_SAT493M if norm_type == "sat" else NORM_IMAGENET
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])
