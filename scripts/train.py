"""
训练入口脚本
用法:
    # 单卡训练
    CUDA_VISIBLE_DEVICES=5 python scripts/train.py --config configs/server.yaml

    # 多卡 DDP 训练 (2卡)
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 scripts/train.py --config configs/server.yaml
"""

import os
import sys
import json
import shutil
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.build import build_model, build_criterion, get_device
from src.datasets.landslide_dataset import create_dataloaders
from src.engine.trainer import Trainer


def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3-LoRA-Mask2Former 滑坡检测训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: cpu/cuda/mps/auto (覆盖配置文件)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数 (覆盖配置文件)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批大小 (覆盖配置文件)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (覆盖配置文件)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (覆盖配置文件)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式: 只用少量数据快速跑通")
    # torchrun will pass this automatically
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()

    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)
    use_ddp = (world_size > 1)

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置
    if args.device:
        config["device"] = args.device
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.output:
        config["output"]["dir"] = args.output

    # 设备
    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = get_device(config)
    if is_main:
        print(f"[Train] Device: {device}, World size: {world_size}")

    # 数据集路径: 相对于项目根目录
    original_root = config["data"]["dataset_root"]
    config["data"]["dataset_root"] = os.path.join(PROJECT_ROOT, original_root)

    # 创建数据加载器
    if is_main:
        print("[Train] Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # DDP: 用 DistributedSampler 替换 train_loader
    train_sampler = None
    if use_ddp:
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            sampler=train_sampler,
            num_workers=config["data"].get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

    if is_main:
        print(f"[Train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 构建模型
    if is_main:
        print("[Train] Building model...")
    model = build_model(config, project_root=PROJECT_ROOT)
    model = model.to(device)

    # DDP: 包装模型
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 优化器 (只优化可训练参数)
    train_cfg = config["training"]
    opt_cfg = train_cfg["optimizer"]
    raw_model = model.module if use_ddp else model
    trainable_params = raw_model.get_trainable_params()
    if is_main:
        print(f"[Train] Trainable parameter groups: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        [p for p in trainable_params],
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    # 学习率调度器
    sched_cfg = train_cfg.get("scheduler", {})
    epochs = train_cfg.get("epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=sched_cfg.get("min_lr", 1e-6),
    )

    # 损失函数 (Mask2Former SetCriterion: Hungarian Matching + CE + Dice + Aux)
    criterion = build_criterion(config)
    criterion = criterion.to(device)
    num_classes = config["model"]["segmentor"].get("num_classes", 2)

    # 输出目录
    output_dir = os.path.join(PROJECT_ROOT, config["output"]["dir"])
    save_every = config["output"].get("save_every", 5)
    log_every = config["output"].get("log_every", 10)

    # 保存配置副本到输出目录 (只在主进程)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(config_path, os.path.join(output_dir, "config.yaml"))

    # 创建训练器
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

    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    # 调试模式: 只跑1个epoch
    if args.debug:
        epochs = 1
        if is_main:
            print("[Train] DEBUG MODE: 1 epoch only")

    # 训练循环
    best_miou = 0.0
    if is_main:
        print(f"\n[Train] Starting training for {epochs} epochs...")
        print(f"[Train] Output dir: {output_dir}")
    t_start = datetime.now()

    for epoch in range(start_epoch, epochs):
        # DDP: 设置 epoch 保证每张卡数据不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练
        train_metrics = trainer.train_one_epoch(train_loader, epoch, log_every)
        if is_main:
            print(f"[Epoch {epoch}/{epochs-1}] loss={train_metrics['loss']:.4f} "
                  f"(CE={train_metrics['ce_loss']:.4f} Dice={train_metrics['dice_loss']:.4f}) "
                  f"lr={train_metrics['lr']:.2e}")

        # 验证 (只在主进程)
        if is_main:
            val_metrics = trainer.validate(val_loader)
            print(f"[Epoch {epoch}/{epochs-1}] mIoU={val_metrics['miou']:.4f} "
                  f"F1={val_metrics['f1']:.4f} "
                  f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}")

            # 记录历史
            trainer.record_epoch(train_metrics, val_metrics)

            # 保存最佳模型
            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                trainer.save_checkpoint(epoch, val_metrics, "best_model.pth")
                print(f"  >> New best mIoU: {best_miou:.4f}")

            # 定期保存
            if (epoch + 1) % save_every == 0:
                trainer.save_checkpoint(epoch, val_metrics)

            # 早停检查
            if trainer.check_early_stopping(best_miou):
                print(f"[Train] Early stopping at epoch {epoch} (no improvement for "
                      f"{trainer.early_stopping_patience} epochs)")
                break

        # DDP: 同步早停信号
        if use_ddp:
            dist.barrier()

    # 保存最终模型和历史 (只在主进程)
    if is_main:
        trainer.save_checkpoint(epoch, val_metrics, "final_model.pth")
        trainer.save_history()
    trainer.close()
    cleanup_ddp()

    if is_main:
        elapsed = datetime.now() - t_start
        print(f"\n[Train] Training complete!")
        print(f"[Train] Best mIoU: {best_miou:.4f}")
        print(f"[Train] Total time: {elapsed}")
        print(f"[Train] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
