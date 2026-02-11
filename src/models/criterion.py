"""
Mask2Former SetCriterion — 官方损失函数移植版
From Mask2Former (Facebook Research), removed detectron2 dependency.
核心: Hungarian Matching + per-query CE + Dice + point sampling + auxiliary loss
Reference: https://github.com/facebookresearch/Mask2Former
"""
import torch
import torch.nn.functional as F
from torch import nn

from .matcher import point_sample


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks.
    Args:
        inputs: (N_matched, H*W) predicted mask logits (will be sigmoided)
        targets: (N_matched, H*W) ground truth binary masks
        num_masks: normalization factor
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    """
    Sigmoid cross entropy loss for masks.
    Args:
        inputs: (N_matched, H*W) predicted mask logits
        targets: (N_matched, H*W) ground truth binary masks
        num_masks: normalization factor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    Estimate uncertainty as L1 distance between 0.0 and the logit prediction.
    Args:
        logits: (R, 1, ...) class-specific logits
    Returns:
        scores: (R, 1, ...) uncertainty scores (higher = more uncertain)
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points based on uncertainty, with some random points mixed in.
    Simplified from detectron2.projects.point_rend.

    Args:
        coarse_logits: (N, 1, H, W) logits
        uncertainty_func: function to compute uncertainty from logits
        num_points: total number of points to sample
        oversample_ratio: oversample factor for importance sampling
        importance_sample_ratio: fraction of points from importance sampling
    Returns:
        point_coords: (N, num_points, 2) normalized coordinates
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)

    # Random point coordinates
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)

    # Sample logits at these points
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    # Calculate uncertainty
    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

    if num_random_points > 0:
        point_coords = torch.cat(
            [point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)],
            dim=1,
        )
    return point_coords


class SetCriterion(nn.Module):
    """
    Mask2Former 的损失函数.
    流程:
        1. Hungarian matching 将 predictions 和 targets 配对
        2. 对配对结果计算 classification loss + mask loss (CE + Dice)
        3. 对每层 decoder 的中间输出也计算 auxiliary loss
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef=0.1,
        losses=("labels", "masks"),
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ):
        """
        Args:
            num_classes: number of object categories (excluding no-object)
            matcher: HungarianMatcher instance
            weight_dict: dict of loss name -> weight, e.g.
                {"loss_ce": 5.0, "loss_mask": 5.0, "loss_dice": 5.0}
            eos_coef: weight for no-object class in classification
            losses: tuple of loss types to compute
            num_points: number of points for point-based mask loss
            oversample_ratio: oversample ratio for uncertain point sampling
            importance_sample_ratio: fraction of importance-sampled points
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # Classification weight: lower weight for no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (Cross Entropy)."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # (B, Q, K+1)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Mask loss: sigmoid CE + Dice, with point sampling."""
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = outputs["pred_masks"].float()  # (B, Q, H, W) — force float32 for AMP safety
        src_masks = src_masks[src_idx]     # (N_matched, H, W)

        # Build flat target mask indices: offset each batch's target indices
        # by the cumulative number of targets in previous batches
        target_masks_list = [t["masks"] for t in targets]
        target_masks = torch.cat(target_masks_list, dim=0)  # (N_total, H, W)
        offsets = [0]
        for t in targets:
            offsets.append(offsets[-1] + len(t["masks"]))
        flat_tgt_idx = torch.cat([
            tgt_j + offsets[i] for i, (_, tgt_j) in enumerate(indices)
        ])
        target_masks = target_masks[flat_tgt_idx]  # (N_matched, H, W)

        # Add channel dim: (N, 1, H, W)
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None].to(src_masks)

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks, point_coords, align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """
        Compute all losses.

        Args:
            outputs: dict with "pred_logits", "pred_masks", and optionally "aux_outputs"
            targets: list of dicts with "labels" and "masks"

        Returns:
            dict of all losses (including auxiliary)
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Hungarian matching on final layer output
        indices = self.matcher(outputs_without_aux, targets)

        # Number of matched masks for normalization
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks, min=1).item()

        # Compute losses on final output
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # Auxiliary losses (intermediate decoder layers)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
