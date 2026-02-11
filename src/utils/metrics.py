"""
评价指标计算
mIoU, F1, Precision, Recall
"""

import numpy as np
import torch


class SegmentationMetrics:
    """语义分割指标累积计算器"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置所有累积量"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        """
        更新混淆矩阵
        
        Args:
            pred: (B, H, W) 预测类别 (int)
            target: (B, H, W) 真实类别 (int)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.astype(np.int64).flatten()
        target = target.astype(np.int64).flatten()

        # 过滤无效值
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        # 更新混淆矩阵
        indices = target * self.num_classes + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        计算所有指标
        
        Returns:
            dict: {miou, f1, precision, recall, iou_per_class, ...}
        """
        cm = self.confusion_matrix

        # IoU per class
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0.0)
        miou = np.mean(iou)

        # Precision, Recall, F1 (针对滑坡类, class=1)
        if self.num_classes >= 2:
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0

        # Overall Accuracy (OA)
        total_correct = np.diag(cm).sum()
        total_pixels = cm.sum()
        oa = total_correct / total_pixels if total_pixels > 0 else 0.0

        # Per-class accuracy
        class_total = cm.sum(axis=1)  # 每类真实像素数
        acc_per_class = np.where(class_total > 0, intersection / class_total, 0.0)

        # Kappa 系数
        pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total_pixels ** 2) if total_pixels > 0 else 0.0
        kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

        return {
            "miou": float(miou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "oa": float(oa),
            "kappa": float(kappa),
            "iou_per_class": iou.tolist(),
            "acc_per_class": acc_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
        }
