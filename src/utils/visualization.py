"""
可视化工具
原图-预测掩膜-边界叠加 对比可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """反归一化 tensor 图像到 0-255 numpy"""
    import torch
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)  # CHW -> HWC
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None, title=None):
    """
    可视化单张预测结果
    
    Args:
        image: (3, H, W) 归一化后的 tensor 或 (H, W, 3) numpy
        pred_mask: (H, W) 预测掩膜 (0/1)
        gt_mask: (H, W) 真实掩膜 (可选)
        save_path: 保存路径
        title: 图标题
    """
    image = denormalize(image) if image.max() <= 1.0 or (hasattr(image, 'shape') and image.shape[0] == 3) else image

    if hasattr(pred_mask, 'cpu'):
        pred_mask = pred_mask.cpu().numpy()
    if gt_mask is not None and hasattr(gt_mask, 'cpu'):
        gt_mask = gt_mask.cpu().numpy()

    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # 预测掩膜叠加
    axes[1].imshow(image)
    axes[1].imshow(pred_mask, alpha=0.4, cmap="Reds")
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    # 真实掩膜 (如果有)
    if gt_mask is not None:
        axes[2].imshow(image)
        axes[2].imshow(gt_mask, alpha=0.4, cmap="Greens")
        axes[2].set_title("Ground Truth Overlay")
        axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_batch(images, pred_masks, gt_masks=None, save_dir=None, prefix="vis"):
    """批量可视化"""
    B = images.shape[0]
    for i in range(B):
        gt = gt_masks[i] if gt_masks is not None else None
        save_path = os.path.join(save_dir, f"{prefix}_{i}.png") if save_dir else None
        visualize_prediction(images[i], pred_masks[i], gt, save_path)
