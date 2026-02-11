"""
Hungarian Matcher for Mask2Former.
From Mask2Former (Facebook Research), removed detectron2 dependency.
Uses scipy.optimize.linear_sum_assignment for Hungarian matching.
Reference: https://github.com/facebookresearch/Mask2Former
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise DICE loss between all predictions and all targets.
    Args:
        inputs: (N, H*W) predicted masks (after sigmoid)
        targets: (M, H*W) ground truth masks
    Returns:
        (N, M) dice loss matrix
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


@torch.jit.script
def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise sigmoid CE loss between all predictions and all targets.
    Args:
        inputs: (N, H*W) predicted masks (logits)
        targets: (M, H*W) ground truth masks
    Returns:
        (N, M) CE loss matrix
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )
    return loss / hw


def point_sample(input_features, point_coords, align_corners=False):
    """
    Sample features at given point coordinates.
    Simplified version of detectron2.projects.point_rend.point_features.point_sample.

    Args:
        input_features: (N, C, H, W) feature map
        point_coords: (N, P, 2) normalized coordinates in [0, 1]
        align_corners: align_corners for grid_sample
    Returns:
        (N, C, P) sampled features
    """
    # grid_sample expects coords in [-1, 1] and shape (N, H_out, W_out, 2)
    # We treat P points as a 1xP grid
    if point_coords.dim() == 3:
        # (N, P, 2) -> (N, 1, P, 2)
        point_coords_grid = point_coords.unsqueeze(1)
    else:
        point_coords_grid = point_coords

    point_coords_grid = 2.0 * point_coords_grid - 1.0
    # Flip x,y to match grid_sample convention (grid_sample expects x,y not y,x)
    output = F.grid_sample(
        input_features, point_coords_grid,
        align_corners=align_corners, mode="bilinear", padding_mode="zeros",
    )
    # (N, C, 1, P) -> (N, C, P)
    if output.shape[2] == 1:
        output = output.squeeze(2)
    return output


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and predictions using Hungarian algorithm.
    For efficiency, uses point sampling on masks instead of full-resolution comparison.
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 12544):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.

        Args:
            outputs: dict with:
                "pred_logits": (B, num_queries, num_classes) classification logits
                "pred_masks": (B, num_queries, H, W) predicted masks
            targets: list of dicts (len = B), each with:
                "labels": (num_targets,) class labels
                "masks": (num_targets, H, W) ground truth masks

        Returns:
            list of (index_i, index_j) tuples for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        # Force float32 — AMP autocast may feed float16 tensors which cause
        # inf in BCE and make the cost matrix infeasible for Hungarian matching.
        device_type = outputs["pred_logits"].device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            for b in range(bs):
                out_prob = outputs["pred_logits"][b].float().softmax(-1)
                tgt_ids = targets[b]["labels"]

                # Classification cost: -prob[target_class]
                cost_class = -out_prob[:, tgt_ids]

                out_mask = outputs["pred_masks"][b].float()  # (num_queries, H, W)
                tgt_mask = targets[b]["masks"].to(out_mask)

                out_mask = out_mask[:, None]  # (num_queries, 1, H, W)
                tgt_mask = tgt_mask[:, None]  # (num_targets, 1, H, W)

                # Point sampling for efficient matching
                point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

                tgt_mask_sampled = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)  # (num_targets, num_points)

                out_mask_sampled = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)  # (num_queries, num_points)

                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)

                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
                # Clamp to prevent inf/NaN which makes linear_sum_assignment fail
                C = C.reshape(num_queries, -1).cpu()
                C = torch.clamp(C, min=-1e8, max=1e8)
                C[C.isnan()] = 1e8
                indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
