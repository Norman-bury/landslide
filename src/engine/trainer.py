"""
训练循环
支持 Mask2Former SetCriterion (Hungarian Matching + CE + Dice + Auxiliary Loss)
以及简单的 SegmentationLoss (用于基线模型)
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
#  简单损失 (用于基线模型 U-Net / DeepLabv3+ / FCN)
# ============================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary/multi-class segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = (probs + targets_onehot).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """CE + Dice 组合损失 (用于基线模型)"""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, num_classes=2):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets.long())
        dice = self.dice_loss(logits, targets)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, {"ce": ce.item(), "dice": dice.item(), "total": total.item()}


# ============================================================
#  Mask2Former 辅助函数
# ============================================================

def prepare_targets(masks, num_classes=2):
    """
    将语义分割 mask 转换为 Mask2Former SetCriterion 期望的 targets 格式.

    输入: masks (B, H, W) 像素级别类别标签 (0=背景, 1=滑坡)
    输出: list of dicts, 每个 dict 包含:
        - "labels": (N_instances,) 类别标签
        - "masks": (N_instances, H, W) 二值 mask

    对于语义分割, 每个类别就是一个 "instance".
    """
    B, H, W = masks.shape
    targets = []
    for b in range(B):
        labels = []
        instance_masks = []
        for c in range(num_classes):
            binary_mask = (masks[b] == c).float()  # (H, W)
            if binary_mask.sum() > 0:  # 只保留非空的类别
                labels.append(c)
                instance_masks.append(binary_mask)
        if len(labels) == 0:
            # 全黑图: 至少保留一个背景
            labels.append(0)
            instance_masks.append(torch.zeros(H, W, device=masks.device))
        targets.append({
            "labels": torch.tensor(labels, dtype=torch.int64, device=masks.device),
            "masks": torch.stack(instance_masks),  # (N, H, W)
        })
    return targets


class Trainer:
    """训练管理器"""

    def __init__(self, model, optimizer, scheduler, criterion, device,
                 grad_accum_steps=1, use_amp=False, output_dir="outputs",
                 early_stopping_patience=0, num_classes=2):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and device.type == "cuda"
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience
        self.num_classes = num_classes

        # 自动检测是否使用 Mask2Former SetCriterion
        from ..models.criterion import SetCriterion
        self.use_mask2former = isinstance(criterion, SetCriterion)

        os.makedirs(output_dir, exist_ok=True)

        # 混合精度 (use torch.amp API, compatible with PyTorch >= 2.0)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tensorboard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
        except ImportError:
            print("[Trainer] tensorboard not installed, skipping TB logging")

        # 训练历史 (完整记录, 方便画图)
        self.history = {
            "epoch": [],
            "train_loss": [], "train_ce_loss": [], "train_mask_loss": [], "train_dice_loss": [],
            "grad_norm": [],
            "lr": [],
            "val_loss": [],
            "val_miou": [], "val_f1": [], "val_precision": [], "val_recall": [],
            "val_oa": [], "val_kappa": [],
            "val_iou_per_class": [], "val_acc_per_class": [],
            "epoch_time": [],
        }

    def _compute_grad_norm(self):
        """计算所有可训练参数的梯度 L2 范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2).item() ** 2
        return total_norm ** 0.5

    def train_one_epoch(self, train_loader, epoch, log_every=10):
        """训练一个 epoch"""
        self.model.train()
        self.criterion.train()
        total_loss = 0.0
        loss_details = {"ce": 0.0, "mask": 0.0, "dice": 0.0}
        num_batches = 0
        total_grad_norm = 0.0
        grad_norm_count = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            if self.use_mask2former:
                # Mask2Former: model returns dict, criterion is SetCriterion
                targets = prepare_targets(masks, num_classes=self.num_classes)
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        losses = self.criterion(outputs, targets)
                        # Weighted sum of all losses
                        loss = sum(
                            losses[k] * self.criterion.weight_dict[k]
                            for k in losses if k in self.criterion.weight_dict
                        )
                        loss = loss / self.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(images)
                    losses = self.criterion(outputs, targets)
                    loss = sum(
                        losses[k] * self.criterion.weight_dict[k]
                        for k in losses if k in self.criterion.weight_dict
                    )
                    loss = loss / self.grad_accum_steps
                    loss.backward()

                total_loss += loss.item() * self.grad_accum_steps
                loss_details["ce"] += losses.get("loss_ce", torch.tensor(0.0)).item()
                loss_details["mask"] += losses.get("loss_mask", torch.tensor(0.0)).item()
                loss_details["dice"] += losses.get("loss_dice", torch.tensor(0.0)).item()
            else:
                # 基线模型: model returns (B, C, H, W), criterion is SegmentationLoss
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images)
                        loss, details = self.criterion(logits, masks)
                        loss = loss / self.grad_accum_steps
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(images)
                    loss, details = self.criterion(logits, masks)
                    loss = loss / self.grad_accum_steps
                    loss.backward()

                total_loss += details["total"]
                loss_details["ce"] += details["ce"]
                loss_details["dice"] += details["dice"]

            # 梯度累积
            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = self._compute_grad_norm()
                total_grad_norm += grad_norm
                grad_norm_count += 1
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            num_batches += 1
            if (step + 1) % log_every == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # Flush remaining accumulated gradients (if last mini-batch didn't complete a full accumulation)
        if (step + 1) % self.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            grad_norm = self._compute_grad_norm()
            total_grad_norm += grad_norm
            grad_norm_count += 1
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        avg_ce = loss_details["ce"] / max(num_batches, 1)
        avg_mask = loss_details["mask"] / max(num_batches, 1)
        avg_dice = loss_details["dice"] / max(num_batches, 1)
        avg_grad_norm = total_grad_norm / max(grad_norm_count, 1)
        epoch_time = time.time() - epoch_start

        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.writer:
            self.writer.add_scalar("train/loss", avg_loss, epoch)
            self.writer.add_scalar("train/ce_loss", avg_ce, epoch)
            self.writer.add_scalar("train/mask_loss", avg_mask, epoch)
            self.writer.add_scalar("train/dice_loss", avg_dice, epoch)
            self.writer.add_scalar("train/grad_norm", avg_grad_norm, epoch)
            self.writer.add_scalar("train/lr", current_lr, epoch)

        return {
            "loss": avg_loss,
            "ce_loss": avg_ce,
            "mask_loss": avg_mask,
            "dice_loss": avg_dice,
            "grad_norm": avg_grad_norm,
            "lr": current_lr,
            "epoch_time": epoch_time,
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        from ..utils.metrics import SegmentationMetrics

        self.model.eval()
        metrics = SegmentationMetrics(num_classes=self.num_classes)
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            if self.use_mask2former:
                outputs = self.model(images)
                # 使用 seg_logits 做评测
                seg_logits = outputs["seg_logits"]  # (B, K, H, W)
                preds = seg_logits.argmax(dim=1)    # (B, H, W)
            else:
                logits = self.model(images)
                preds = logits.argmax(dim=1)

            metrics.update(preds, masks)
            num_batches += 1

        results = metrics.compute()

        if self.writer:
            ep = len(self.history["val_miou"])
            self.writer.add_scalar("val/miou", results["miou"], ep)
            self.writer.add_scalar("val/f1", results["f1"], ep)
            self.writer.add_scalar("val/precision", results["precision"], ep)
            self.writer.add_scalar("val/recall", results["recall"], ep)
            self.writer.add_scalar("val/oa", results["oa"], ep)
            self.writer.add_scalar("val/kappa", results["kappa"], ep)

        return results

    def record_epoch(self, train_metrics, val_metrics):
        """记录一个 epoch 的全部指标到历史, 并自动保存 JSON"""
        self.history["epoch"].append(len(self.history["epoch"]))
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["train_ce_loss"].append(train_metrics.get("ce_loss", 0))
        self.history["train_mask_loss"].append(train_metrics.get("mask_loss", 0))
        self.history["train_dice_loss"].append(train_metrics.get("dice_loss", 0))
        self.history["grad_norm"].append(train_metrics.get("grad_norm", 0))
        self.history["lr"].append(train_metrics.get("lr", 0))
        self.history["val_miou"].append(val_metrics["miou"])
        self.history["val_f1"].append(val_metrics["f1"])
        self.history["val_precision"].append(val_metrics["precision"])
        self.history["val_recall"].append(val_metrics["recall"])
        self.history["val_oa"].append(val_metrics.get("oa", 0))
        self.history["val_kappa"].append(val_metrics.get("kappa", 0))
        self.history["val_iou_per_class"].append(val_metrics.get("iou_per_class", []))
        self.history["val_acc_per_class"].append(val_metrics.get("acc_per_class", []))
        self.history["epoch_time"].append(train_metrics.get("epoch_time", 0))
        # 每 epoch 自动保存, 防止中断丢失
        self.save_history()

    def check_early_stopping(self, best_miou):
        """检查是否需要早停"""
        if self.early_stopping_patience <= 0:
            return False
        mious = self.history["val_miou"]
        if len(mious) < self.early_stopping_patience:
            return False
        recent = mious[-self.early_stopping_patience:]
        if all(m <= best_miou for m in recent) and max(recent) < best_miou + 1e-6:
            return True
        return False

    def save_history(self):
        """保存训练历史到 JSON"""
        path = os.path.join(self.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(self, epoch, metrics=None, filename=None):
        """保存 checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch{epoch:03d}.pth"
        path = os.path.join(self.output_dir, filename)

        # DDP: save unwrapped model state_dict
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if metrics is not None:
            state["metrics"] = metrics

        torch.save(state, path)
        print(f"[Trainer] Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path):
        """加载 checkpoint"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        print(f"[Trainer] Checkpoint loaded: {path}, epoch={state.get('epoch', '?')}")
        return state.get("epoch", 0)

    def close(self):
        """关闭 tensorboard writer"""
        if self.writer:
            self.writer.close()
