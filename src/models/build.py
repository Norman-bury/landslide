"""
模型构建入口
根据配置文件创建完整的 DINOv3-LoRA-Mask2Former 模型
"""

import os
import torch
from .backbone import DINOv3Backbone
from .segmentor_v2 import DINOv3Mask2Former


def build_model(config, project_root=None):
    """
    根据配置构建模型
    
    Args:
        config: dict, 从 yaml 加载的配置
        project_root: 项目根目录 (用于解析本地模型路径)
    
    Returns:
        model: DINOv3Mask2Former 实例
    """
    model_cfg = config["model"]
    backbone_cfg = model_cfg["backbone"]
    lora_cfg = model_cfg.get("lora", {})
    seg_cfg = model_cfg.get("segmentor", {})

    # 解析模型路径
    model_name = backbone_cfg["name"]
    if not model_name.startswith("facebook/") and project_root:
        # 本地路径: 相对于项目根目录
        local_path = os.path.join(project_root, model_name)
        if os.path.exists(local_path):
            model_name = local_path
            print(f"[build_model] Using local model: {local_path}")

    # 确定 out_indices
    out_indices = backbone_cfg.get("out_indices", None)

    # 构建 backbone
    backbone = DINOv3Backbone(
        model_name_or_path=model_name,
        freeze=backbone_cfg.get("freeze", True),
        out_indices=out_indices,
        lora_config=lora_cfg,
    )

    # 如果 out_indices 未指定, 根据实际层数自动设置
    if out_indices is not None:
        # 修正: ViT-S 只有12层, 不能用 [5,11,17,23]
        valid_indices = [i for i in out_indices if i < backbone.num_layers]
        if len(valid_indices) < len(out_indices):
            print(f"[build_model] WARNING: adjusted out_indices from {out_indices} to {valid_indices}")
            backbone.out_indices = valid_indices

    # 构建完整模型
    model = DINOv3Mask2Former(
        backbone=backbone,
        num_classes=seg_cfg.get("num_classes", 2),
        hidden_dim=seg_cfg.get("hidden_dim", 256),
        num_queries=seg_cfg.get("num_queries", 100),
        num_decoder_layers=seg_cfg.get("num_decoder_layers", 9),
        nheads=seg_cfg.get("nheads", 8),
        dim_feedforward=seg_cfg.get("dim_feedforward", 2048),
        mask_dim=seg_cfg.get("mask_dim", 256),
        pre_norm=seg_cfg.get("pre_norm", False),
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_model] Total params: {total_params:,}")
    print(f"[build_model] Trainable params: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model


def build_criterion(config):
    """
    构建 Mask2Former 的 SetCriterion (Hungarian Matching + CE + Dice).

    Args:
        config: dict, 从 yaml 加载的配置
    Returns:
        criterion: SetCriterion 实例
    """
    from .matcher import HungarianMatcher
    from .criterion import SetCriterion

    seg_cfg = config["model"].get("segmentor", {})
    loss_cfg = config["training"].get("loss", {})

    num_classes = seg_cfg.get("num_classes", 2)

    matcher = HungarianMatcher(
        cost_class=loss_cfg.get("cost_class", 2.0),
        cost_mask=loss_cfg.get("cost_mask", 5.0),
        cost_dice=loss_cfg.get("cost_dice", 5.0),
        num_points=loss_cfg.get("num_points", 12544),
    )

    weight_dict = {
        "loss_ce": loss_cfg.get("ce_weight", 5.0),
        "loss_mask": loss_cfg.get("mask_weight", 5.0),
        "loss_dice": loss_cfg.get("dice_weight", 5.0),
    }
    # Auxiliary loss weights (same weight for each decoder layer)
    dec_layers = seg_cfg.get("num_decoder_layers", 9)
    aux_weight_dict = {}
    for i in range(dec_layers):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=loss_cfg.get("eos_coef", 0.1),
        losses=("labels", "masks"),
        num_points=loss_cfg.get("num_points", 12544),
        oversample_ratio=loss_cfg.get("oversample_ratio", 3.0),
        importance_sample_ratio=loss_cfg.get("importance_sample_ratio", 0.75),
    )

    return criterion


def get_device(config):
    """根据配置获取设备"""
    device_str = config.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)
