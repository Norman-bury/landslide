"""
统一滑坡数据集类
支持 Bijie (PNG) 和 Moxizhen (TIFF) 两种格式
用于语义分割任务: 2类 (背景=0, 滑坡=1)
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class BijieDataset(Dataset):
    """
    Bijie 滑坡数据集 (PNG格式, 卫星遥感)
    - landslide/image/*.png + landslide/mask/*.png (滑坡样本, 有mask)
    - non-landslide/image/*.png (非滑坡样本, mask全0)
    """

    def __init__(self, root, split="train", train_ratio=0.8, transform=None, seed=42):
        """
        Args:
            root: Bijie-landslide-dataset 根目录
            split: "train" 或 "val"
            train_ratio: 训练集比例
            transform: albumentations 变换
            seed: 随机种子 (保证划分一致)
        """
        self.root = root
        self.split = split
        self.transform = transform

        # 收集滑坡样本 (有image和mask)
        landslide_img_dir = os.path.join(root, "landslide", "image")
        landslide_mask_dir = os.path.join(root, "landslide", "mask")
        landslide_imgs = sorted(glob.glob(os.path.join(landslide_img_dir, "*.png")))

        self.samples = []
        for img_path in landslide_imgs:
            fname = os.path.basename(img_path)
            mask_path = os.path.join(landslide_mask_dir, fname)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path, True))

        # 收集非滑坡样本 (只有image, mask全0)
        non_landslide_img_dir = os.path.join(root, "non-landslide", "image")
        non_landslide_imgs = sorted(glob.glob(os.path.join(non_landslide_img_dir, "*.png")))
        for img_path in non_landslide_imgs:
            self.samples.append((img_path, None, False))

        # 按固定种子划分 train/val
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:n_train]
        else:
            selected = indices[n_train:]

        self.samples = [self.samples[i] for i in selected]
        print(f"[BijieDataset] split={split}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, has_landslide = self.samples[idx]

        # 读取RGB图像
        image = np.array(Image.open(img_path).convert("RGB"))

        # 读取或生成mask
        if has_landslide and mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L"))
            # 二值化: >0 为滑坡(1), 否则为背景(0)
            mask = (mask > 0).astype(np.int64)
        else:
            # 非滑坡样本: 全0 mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]       # (C, H, W) float tensor
            mask = transformed["mask"]          # (H, W) long tensor

        return {"image": image, "mask": mask, "path": img_path}


class MoxizhenDataset(Dataset):
    """
    Moxizhen 滑坡数据集 (TIFF格式, 无人机)
    使用 label 文件夹作为标注 (非 mask)
    """

    def __init__(self, root, transform=None):
        """
        Args:
            root: moxizhen/moxizhen 根目录
            transform: albumentations 变换
        """
        self.root = root
        self.transform = transform

        img_dir = os.path.join(root, "img")
        label_dir = os.path.join(root, "label")

        # 收集配对的 img + label
        self.samples = []
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        for img_path in img_files:
            fname = os.path.basename(img_path)
            label_path = os.path.join(label_dir, fname)
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))

        print(f"[MoxizhenDataset] samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        if HAS_RASTERIO:
            image = self._read_tiff_rasterio(img_path)
            label = self._read_tiff_rasterio(label_path, is_label=True)
        else:
            # fallback: PIL (可能丢失地理信息, 但能读取像素)
            image = np.array(Image.open(img_path).convert("RGB"))
            label_img = np.array(Image.open(label_path))
            label = (label_img > 0).astype(np.int64)
            if label.ndim == 3:
                label = label[:, :, 0]

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        return {"image": image, "mask": label, "path": img_path}

    @staticmethod
    def _read_tiff_rasterio(path, is_label=False):
        """使用 rasterio 读取 TIFF 文件"""
        with rasterio.open(path) as src:
            if is_label:
                data = src.read(1)  # 单波段
                return (data > 0).astype(np.int64)
            else:
                # 读取前3个波段 (RGB)
                n_bands = min(src.count, 3)
                bands = [src.read(i + 1) for i in range(n_bands)]
                if n_bands == 1:
                    image = np.stack([bands[0]] * 3, axis=-1)
                elif n_bands == 2:
                    image = np.stack([bands[0], bands[1], bands[0]], axis=-1)
                else:
                    image = np.stack(bands, axis=-1)
                # 归一化到 0-255 uint8
                if image.dtype != np.uint8:
                    img_min, img_max = image.min(), image.max()
                    if img_max > img_min:
                        image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)
                return image


def create_dataloaders(config):
    """
    根据配置创建 DataLoader
    
    Args:
        config: dict, 从 yaml 加载的配置
    
    Returns:
        train_loader, val_loader, test_loader (Moxizhen)
    """
    from .transforms import get_train_transforms, get_val_transforms

    data_cfg = config["data"]
    dataset_root = data_cfg["dataset_root"]
    image_size = data_cfg.get("image_size", 512)
    batch_size = data_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 0)
    norm_type = config.get("model", {}).get("backbone", {}).get("norm_type", "imagenet")
    pin_memory = config.get("device", "cpu") != "cpu"

    # Bijie 训练集和验证集
    bijie_cfg = data_cfg["bijie"]
    bijie_root = os.path.join(dataset_root, bijie_cfg["root"])

    train_dataset = BijieDataset(
        root=bijie_root,
        split="train",
        train_ratio=bijie_cfg.get("train_ratio", 0.8),
        transform=get_train_transforms(image_size, norm_type=norm_type),
    )
    val_dataset = BijieDataset(
        root=bijie_root,
        split="val",
        train_ratio=bijie_cfg.get("train_ratio", 0.8),
        transform=get_val_transforms(image_size, norm_type=norm_type),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # Moxizhen 测试集 (跨域零样本)
    mox_cfg = data_cfg["moxizhen"]
    mox_root = os.path.join(dataset_root, mox_cfg["root"])

    test_dataset = MoxizhenDataset(
        root=mox_root,
        transform=get_val_transforms(image_size, norm_type=norm_type),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
