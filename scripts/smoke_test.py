"""
Smoke test: 验证整个 DINOv3-LoRA-Mask2Former 流程能在 CPU 上跑通
不需要真实训练，只验证:
1. 数据集加载
2. 模型构建 (DINOv3 + LoRA + Mask2Former decoder)
3. 前向传播
4. 损失计算
5. 反向传播
6. 指标计算
"""

import os
import sys
import time
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_step(name, func):
    """运行一个测试步骤"""
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = func()
        dt = time.time() - t0
        print(f"  ✓ PASS ({dt:.1f}s)")
        return result
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✗ FAIL ({dt:.1f}s): {e}")
        import traceback
        traceback.print_exc()
        return None


def step1_load_config():
    """加载配置"""
    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 强制 CPU + 小参数加速测试
    config["device"] = "cpu"
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["model"]["segmentor"]["num_queries"] = 10
    config["model"]["segmentor"]["num_decoder_layers"] = 3
    config["model"]["segmentor"]["dim_feedforward"] = 512
    config["training"]["loss"]["num_points"] = 256
    
    print(f"  Model: {config['model']['backbone']['name']}")
    print(f"  Device: {config['device']}")
    return config


def step2_load_dataset(config):
    """测试数据集加载"""
    from src.datasets.landslide_dataset import BijieDataset
    from src.datasets.transforms import get_val_transforms

    data_cfg = config["data"]
    bijie_root = os.path.join(
        PROJECT_ROOT, data_cfg["dataset_root"],
        data_cfg["bijie"]["root"]
    )
    
    transform = get_val_transforms(image_size=224)  # 用小图加速
    dataset = BijieDataset(
        root=bijie_root,
        split="train",
        train_ratio=0.8,
        transform=transform,
    )
    
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Mask unique values: {sample['mask'].unique().tolist()}")
    return dataset, sample


def step3_build_model(config):
    """测试模型构建"""
    from src.models.build import build_model
    model = build_model(config, project_root=PROJECT_ROOT)
    model.eval()
    return model


def step4_forward(model, sample):
    """测试前向传播 (Mask2Former dict output)"""
    image = sample["image"].unsqueeze(0)  # (1, 3, H, W)
    print(f"  Input shape: {image.shape}")
    
    with torch.no_grad():
        outputs = model(image)
    
    print(f"  pred_logits: {outputs['pred_logits'].shape}")
    print(f"  pred_masks: {outputs['pred_masks'].shape}")
    print(f"  seg_logits: {outputs['seg_logits'].shape}")
    print(f"  aux_outputs: {len(outputs['aux_outputs'])} layers")
    
    preds = outputs['seg_logits'].argmax(dim=1)
    print(f"  Predictions: {preds.shape}")
    return outputs


def step5_loss(output, sample):
    """测试损失计算"""
    from src.engine.trainer import SegmentationLoss
    
    criterion = SegmentationLoss(ce_weight=1.0, dice_weight=1.0)
    mask = sample["mask"].unsqueeze(0)  # (1, H, W)
    
    # output 可能和 mask 尺寸不同，需要确保一致
    if output.shape[2:] != mask.shape[1:]:
        import torch.nn.functional as F
        mask = F.interpolate(
            mask.unsqueeze(1).float(), size=output.shape[2:], mode="nearest"
        ).squeeze(1).long()
    
    loss, details = criterion(output, mask)
    print(f"  Total loss: {details['total']:.4f}")
    print(f"  CE loss: {details['ce']:.4f}")
    print(f"  Dice loss: {details['dice']:.4f}")
    return loss


def step6_backward(model, loss):
    """测试反向传播"""
    # 需要重新做一次有梯度的前向传播
    loss.backward()
    
    # 检查梯度
    grad_params = 0
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_params += 1
            total_grad_norm += param.grad.norm().item() ** 2
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Parameters with gradients: {grad_params}")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    return True


def step7_metrics():
    """测试指标计算"""
    from src.utils.metrics import SegmentationMetrics
    
    metrics = SegmentationMetrics(num_classes=2)
    
    # 模拟预测
    pred = torch.zeros(2, 64, 64, dtype=torch.long)
    pred[0, 10:30, 10:30] = 1
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[0, 15:35, 15:35] = 1
    
    metrics.update(pred, target)
    results = metrics.compute()
    
    print(f"  mIoU: {results['miou']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    return results


def main():
    print("\n" + "=" * 60)
    print("  DINOv3-LoRA-Mask2Former Smoke Test")
    print("=" * 60)
    
    # Step 1: Config
    config = test_step("Load Config", step1_load_config)
    if config is None:
        return
    
    # Step 2: Dataset
    result = test_step("Load Dataset", lambda: step2_load_dataset(config))
    if result is None:
        return
    dataset, sample = result
    
    # Step 3: Build Model
    model = test_step("Build Model (DINOv3 + LoRA + Mask2Former)", 
                       lambda: step3_build_model(config))
    if model is None:
        return
    
    # Step 4: Forward pass
    output = test_step("Forward Pass", lambda: step4_forward(model, sample))
    if output is None:
        return
    
    # Step 5: Loss + Backward (Mask2Former SetCriterion)
    print(f"\n{'='*60}")
    print(f"  Step: SetCriterion Loss + Backward")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        from src.models.build import build_criterion
        from src.engine.trainer import prepare_targets
        
        model.train()
        criterion = build_criterion(config)
        
        image = sample["image"].unsqueeze(0)
        mask = sample["mask"].unsqueeze(0)
        
        outputs = model(image)
        targets = prepare_targets(mask, num_classes=2)
        
        losses = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        total_loss = sum(losses[k] * weight_dict[k] for k in losses if k in weight_dict)
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  loss_ce: {losses['loss_ce'].item():.4f}")
        print(f"  loss_mask: {losses['loss_mask'].item():.4f}")
        print(f"  loss_dice: {losses['loss_dice'].item():.4f}")
        print(f"  Num losses (incl. aux): {len(losses)}")
        
        total_loss.backward()
        
        grad_params = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        print(f"  Parameters with gradients: {grad_params}")
        
        dt = time.time() - t0
        print(f"  ✓ PASS ({dt:.1f}s)")
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✗ FAIL ({dt:.1f}s): {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Metrics
    test_step("Metrics Calculation", step7_metrics)
    
    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
