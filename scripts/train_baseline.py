"""
对比实验: 训练基线模型 (U-Net, DeepLabv3+, FCN)
用法:
    python scripts/train_baseline.py --model unet
    python scripts/train_baseline.py --model deeplabv3plus
    python scripts/train_baseline.py --model fcn
"""

import os
import sys
import json
import shutil
import argparse
import yaml
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.baselines import build_baseline
from src.models.build import get_device
from src.datasets.landslide_dataset import create_dataloaders
from src.engine.trainer import Trainer, SegmentationLoss
from src.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="基线模型对比实验")
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "deeplabv3plus", "fcn"],
                        help="基线模型名称")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    device = get_device(config)
    print(f"[Baseline] Model: {args.model}, Device: {device}")

    # 数据
    config["data"]["dataset_root"] = os.path.join(PROJECT_ROOT, config["data"]["dataset_root"])
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # 构建基线模型
    num_classes = config["model"]["segmentor"]["num_classes"]
    model = build_baseline(args.model, num_classes=num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Baseline] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 优化器
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=train_cfg["optimizer"].get("weight_decay", 0.01),
    )

    epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    # 基线模型使用简单 CE+Dice, 权重固定为 1.0 (不用 Mask2Former 的 5.0)
    criterion = SegmentationLoss(ce_weight=1.0, dice_weight=1.0)

    # 输出目录
    output_dir = os.path.join(PROJECT_ROOT, "outputs", f"baseline_{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        grad_accum_steps=train_cfg.get("gradient_accumulation_steps", 1),
        use_amp=train_cfg.get("amp", False),
        output_dir=output_dir,
        early_stopping_patience=train_cfg.get("early_stopping_patience", 0),
    )

    # 训练
    best_miou = 0.0
    log_every = config["output"].get("log_every", 10)
    save_every = config["output"].get("save_every", 10)
    t_start = datetime.now()

    print(f"\n[Baseline] Training {args.model} for {epochs} epochs...")
    for epoch in range(epochs):
        train_metrics = trainer.train_one_epoch(train_loader, epoch, log_every)
        val_metrics = trainer.validate(val_loader)
        trainer.record_epoch(train_metrics, val_metrics)

        print(f"[Epoch {epoch}/{epochs-1}] loss={train_metrics['loss']:.4f} "
              f"mIoU={val_metrics['miou']:.4f} F1={val_metrics['f1']:.4f}")

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            trainer.save_checkpoint(epoch, val_metrics, "best_model.pth")

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch, val_metrics)

        if trainer.check_early_stopping(best_miou):
            print(f"[Baseline] Early stopping at epoch {epoch}")
            break

    trainer.save_checkpoint(epoch, val_metrics, "final_model.pth")
    trainer.save_history()

    # 评测
    print(f"\n[Baseline] Evaluating {args.model}...")
    evaluator = Evaluator(model, device, num_classes=num_classes)

    all_results = {
        "model": args.model,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_miou": best_miou,
        "training_time": str(datetime.now() - t_start),
    }

    print("\n--- Bijie Validation ---")
    val_results = evaluator.evaluate(val_loader, desc="Eval Val")
    evaluator.print_results(val_results, f"{args.model} - Bijie Val")
    all_results["bijie_val"] = {k: v for k, v in val_results.items() if k != "confusion_matrix"}

    print("\n--- Moxizhen Cross-Domain ---")
    test_results = evaluator.evaluate(test_loader, desc="Eval Moxizhen")
    evaluator.print_results(test_results, f"{args.model} - Moxizhen")
    all_results["moxizhen_test"] = {k: v for k, v in test_results.items() if k != "confusion_matrix"}

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    trainer.close()
    elapsed = datetime.now() - t_start
    print(f"\n[Baseline] {args.model} complete! Best mIoU: {best_miou:.4f}, Time: {elapsed}")


if __name__ == "__main__":
    main()
