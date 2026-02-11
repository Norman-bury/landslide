"""
评测管线
零样本跨域评测: 在目标域数据上评估 mIoU, F1, Precision, Recall
"""

import os
import torch
from tqdm import tqdm
from ..utils.metrics import SegmentationMetrics


class Evaluator:
    """跨域评测器"""

    def __init__(self, model, device, num_classes=2):
        self.model = model
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self, dataloader, desc="Evaluating"):
        """
        在给定数据集上评测
        
        Args:
            dataloader: DataLoader
            desc: 进度条描述
        
        Returns:
            dict: {miou, f1, precision, recall, ...}
        """
        self.model.eval()
        metrics = SegmentationMetrics(self.num_classes)

        for batch in tqdm(dataloader, desc=desc):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            outputs = self.model(images)
            # Mask2Former returns dict with "seg_logits", baselines return tensor
            if isinstance(outputs, dict):
                logits = outputs["seg_logits"]
            else:
                logits = outputs
            preds = logits.argmax(dim=1)

            metrics.update(preds, masks)

        results = metrics.compute()
        return results

    def print_results(self, results, title="Evaluation Results"):
        """打印评测结果"""
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
        print(f"  mIoU:      {results['miou']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        if "iou_per_class" in results:
            for i, iou in enumerate(results["iou_per_class"]):
                cls_name = "Background" if i == 0 else "Landslide"
                print(f"  IoU ({cls_name}): {iou:.4f}")
        print(f"{'='*50}\n")
