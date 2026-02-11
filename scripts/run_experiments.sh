#!/bin/bash
# =============================================================
# 完整实验流程: 主模型 + 基线对比 + 消融实验
# 用法: bash scripts/run_experiments.sh
# =============================================================

set -e
echo "=========================================="
echo "  DINOv3-LoRA-Mask2Former 完整实验"
echo "=========================================="

CONFIG="configs/server.yaml"

# ----- 1. 主模型训练 -----
echo ""
echo ">>> [1/6] 训练主模型: DINOv3-LoRA-Mask2Former"
python scripts/train.py --config $CONFIG

echo ""
echo ">>> [1/6] 评测主模型"
python scripts/evaluate.py --config $CONFIG --checkpoint outputs/best_model.pth

# ----- 2. 基线对比 -----
echo ""
echo ">>> [2/6] 训练基线: U-Net"
python scripts/train_baseline.py --model unet --config $CONFIG

echo ""
echo ">>> [3/6] 训练基线: DeepLabv3+"
python scripts/train_baseline.py --model deeplabv3plus --config $CONFIG

echo ""
echo ">>> [4/6] 训练基线: FCN"
python scripts/train_baseline.py --model fcn --config $CONFIG

# ----- 3. 消融实验 -----
echo ""
echo ">>> [5/6] 消融实验: 不同 LoRA rank"
for RANK in 4 8 16 32; do
    echo "  -- LoRA rank=$RANK"
    python scripts/train_ablation.py --ablation lora_rank --value $RANK --config $CONFIG
done

echo ""
echo ">>> [6/8] 消融实验: LoRA 插入层数"
for N in 2 4 6 8; do
    echo "  -- LoRA last $N layers"
    python scripts/train_ablation.py --ablation lora_layers --value $N --config $CONFIG
done

echo ""
echo ">>> [7/8] 消融实验: 无 LoRA (全冻结 backbone)"
python scripts/train_ablation.py --ablation no_lora --config $CONFIG

echo ""
echo ">>> [8/8] 消融实验: 无 Dice loss"
python scripts/train_ablation.py --ablation no_dice --config $CONFIG

# ----- 4. 汇总结果 -----
echo ""
echo ">>> 汇总所有实验结果"
python scripts/summarize_results.py

echo ""
echo "=========================================="
echo "  所有实验完成!"
echo "=========================================="
