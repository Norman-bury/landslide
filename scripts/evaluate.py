"""
评测入口脚本
用法:
    python scripts/evaluate.py --checkpoint outputs/best_model.pth
    python scripts/evaluate.py --checkpoint outputs/best_model.pth --dataset moxizhen
    python scripts/evaluate.py --checkpoint outputs/best_model.pth --output eval_results/
"""

import os
import sys
import json
import argparse
import yaml
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.build import build_model, get_device
from src.datasets.landslide_dataset import create_dataloaders
from src.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3-LoRA-Mask2Former 滑坡检测评测")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["val", "moxizhen", "all"],
                        help="评测数据集: val(Bijie验证集), moxizhen(跨域), all")
    parser.add_argument("--output", type=str, default=None,
                        help="评测结果输出目录 (默认: checkpoint同目录)")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device

    device = get_device(config)
    print(f"[Eval] Device: {device}")

    # 数据
    config["data"]["dataset_root"] = os.path.join(PROJECT_ROOT, config["data"]["dataset_root"])
    _, val_loader, test_loader = create_dataloaders(config)

    # 模型
    model = build_model(config, project_root=PROJECT_ROOT)

    # 加载 checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    print(f"[Eval] Loaded checkpoint: {ckpt_path} (epoch {state.get('epoch', '?')})")

    # 输出目录
    if args.output:
        output_dir = os.path.join(PROJECT_ROOT, args.output)
    else:
        output_dir = os.path.dirname(ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    num_classes = config["model"]["segmentor"].get("num_classes", 2)
    evaluator = Evaluator(model, device, num_classes=num_classes)
    all_results = {
        "checkpoint": ckpt_path,
        "epoch": state.get("epoch", None),
        "timestamp": datetime.now().isoformat(),
    }

    # 评测
    if args.dataset in ("val", "all"):
        print("\n--- Bijie Validation Set (Source Domain) ---")
        results = evaluator.evaluate(val_loader, desc="Eval Bijie Val")
        evaluator.print_results(results, "Bijie Validation (Source Domain)")
        all_results["bijie_val"] = {
            k: v for k, v in results.items() if k != "confusion_matrix"
        }

    if args.dataset in ("moxizhen", "all"):
        print("\n--- Moxizhen Test Set (Cross-Domain Zero-Shot) ---")
        results = evaluator.evaluate(test_loader, desc="Eval Moxizhen")
        evaluator.print_results(results, "Moxizhen (Cross-Domain Zero-Shot)")
        all_results["moxizhen_test"] = {
            k: v for k, v in results.items() if k != "confusion_matrix"
        }

    # 保存结果到 JSON
    result_path = os.path.join(output_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results saved to: {result_path}")


if __name__ == "__main__":
    main()
