"""
消融实验脚本
支持:
  - lora_rank: 不同 LoRA rank (4, 8, 16, 32)
  - no_lora: 不使用 LoRA (全冻结 backbone, 只训练 decoder)
  - no_dice: 不使用 Dice loss (只用 CE)
  - no_fpn: 单尺度特征 (不用 FPN 多尺度)

用法:
    python scripts/train_ablation.py --ablation lora_rank --value 4
    python scripts/train_ablation.py --ablation no_lora
    python scripts/train_ablation.py --ablation no_dice
"""

import os
import sys
import json
import copy
import argparse
import yaml
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.build import build_model, build_criterion, get_device
from src.datasets.landslide_dataset import create_dataloaders
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["lora_rank", "no_lora", "no_dice", "no_fpn", "lora_layers"],
                        help="消融类型")
    parser.add_argument("--value", type=str, default=None,
                        help="消融参数值 (如 lora_rank 的 rank 值)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def apply_ablation(config, ablation, value):
    """根据消融类型修改配置"""
    config = copy.deepcopy(config)
    exp_name = ablation

    if ablation == "lora_rank":
        rank = int(value)
        config["model"]["lora"]["rank"] = rank
        config["model"]["lora"]["alpha"] = rank * 2
        exp_name = f"lora_rank{rank}"

    elif ablation == "no_lora":
        config["model"]["lora"]["enabled"] = False
        exp_name = "no_lora"

    elif ablation == "no_dice":
        config["training"]["loss"]["dice_weight"] = 0.0
        exp_name = "no_dice"

    elif ablation == "lora_layers":
        # value = 插入LoRA的层数 (从尾部算), 如 value=3 表示最后3层
        n = int(value)
        num_layers = 24  # ViT-L
        layers = list(range(num_layers - n, num_layers))
        config["model"]["lora"]["lora_layers"] = layers
        exp_name = f"lora_last{n}layers"

    elif ablation == "no_fpn":
        # 减少到2层特征 (FPN 最少需要2层, 不能只用1层)
        out_indices = config["model"]["backbone"].get("out_indices", [5, 11, 17, 23])
        config["model"]["backbone"]["out_indices"] = out_indices[-2:]
        exp_name = "reduced_scale"

    return config, exp_name


def main():
    args = parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    # 应用消融修改
    config, exp_name = apply_ablation(config, args.ablation, args.value)

    device = get_device(config)
    print(f"[Ablation] Experiment: {exp_name}, Device: {device}")

    # 数据
    config["data"]["dataset_root"] = os.path.join(PROJECT_ROOT, config["data"]["dataset_root"])
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # 模型
    model = build_model(config, project_root=PROJECT_ROOT)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Ablation] Params: total={total_params:,}, trainable={trainable_params:,}")

    # 优化器
    train_cfg = config["training"]
    trainable = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=train_cfg["optimizer"].get("lr", 1e-4),
        weight_decay=train_cfg["optimizer"].get("weight_decay", 0.01),
    )

    epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )

    criterion = build_criterion(config)
    criterion = criterion.to(device)
    num_classes = config["model"]["segmentor"].get("num_classes", 2)

    output_dir = os.path.join(PROJECT_ROOT, "outputs", f"ablation_{exp_name}")
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
        num_classes=num_classes,
    )

    # 训练
    best_miou = 0.0
    log_every = config["output"].get("log_every", 10)
    save_every = config["output"].get("save_every", 10)
    t_start = datetime.now()

    for epoch in range(epochs):
        train_metrics = trainer.train_one_epoch(train_loader, epoch, log_every)
        val_metrics = trainer.validate(val_loader)
        trainer.record_epoch(train_metrics, val_metrics)

        print(f"[{exp_name} Epoch {epoch}/{epochs-1}] loss={train_metrics['loss']:.4f} "
              f"mIoU={val_metrics['miou']:.4f} F1={val_metrics['f1']:.4f}")

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            trainer.save_checkpoint(epoch, val_metrics, "best_model.pth")

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch, val_metrics)

        if trainer.check_early_stopping(best_miou):
            print(f"[Ablation] Early stopping at epoch {epoch}")
            break

    trainer.save_checkpoint(epoch, val_metrics, "final_model.pth")
    trainer.save_history()

    # 评测
    evaluator = Evaluator(model, device, num_classes=config["model"]["segmentor"]["num_classes"])
    all_results = {
        "experiment": exp_name,
        "ablation": args.ablation,
        "value": args.value,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_miou": best_miou,
        "training_time": str(datetime.now() - t_start),
    }

    val_results = evaluator.evaluate(val_loader, desc="Eval Val")
    all_results["bijie_val"] = {k: v for k, v in val_results.items() if k != "confusion_matrix"}

    test_results = evaluator.evaluate(test_loader, desc="Eval Moxizhen")
    all_results["moxizhen_test"] = {k: v for k, v in test_results.items() if k != "confusion_matrix"}

    evaluator.print_results(val_results, f"{exp_name} - Bijie Val")
    evaluator.print_results(test_results, f"{exp_name} - Moxizhen")

    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    trainer.close()
    print(f"\n[Ablation] {exp_name} done! Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
