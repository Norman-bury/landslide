# 滑坡检测项目 - DINOv3-LoRA-Mask2Former

## 项目概述
基于开题报告，实现 DINOv3-LoRA-Mask2Former 滑坡语义分割模型，重点验证跨平台、跨传感器、跨区域泛化能力。

## 关键决策
- **不使用 DEM 数据**，所有输入统一为 RGB 3通道
- **Moxizhen 使用 label 文件夹**（非 mask），label 记录滑坡区域信息
- **Moxizhen v2.0 已删除**，暂不使用
- **训练源域**：Bijie (SAT卫星数据, PNG格式)
- **跨域测试**：Moxizhen v1 (UAV无人机数据, TIFF格式)
- **本地 Mac 无 GPU**：CPU + 小 batch 跑通流程，服务器上正式训练
- **本地调试模型**：facebook/dinov3-vits16-pretrain-lvd1689m (ViT-S/16, 384维, 21M参数, gated model 需网页申请访问权限)
- **服务器训练模型**：dinov3-vitl16-pretrain-sat493m (ViT-L/16, 1024维, 300M参数, 卫星预训练, 已下载)
- **DINOv3 vs DINOv2**：DINOv3 使用 RoPE 位置编码 + 4个 register tokens，需要 transformers>=4.56.0
- **卫星模型归一化**：mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)

## 目录结构
```
d_project/
├── readme.md                          # 本文件 - 项目进度与计划
├── 开题报告.md                           # 研究提案
├── configs/                             # 配置文件
│   ├── default.yaml                     # 默认训练/评测配置
│   └── server.yaml                      # 服务器训练配置 (A100-80GB)
├── src/                                 # 核心源代码
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                  # DINOv3 + LoRA 封装
│   │   ├── baselines.py                 # 基线模型 (U-Net, DeepLabv3+, FCN)
│   │   ├── build.py                     # 模型构建入口
│   │   ├── criterion.py                 # Mask2Former SetCriterion (Hungarian Matching)
│   │   ├── matcher.py                   # Hungarian Matcher + point sampling
│   │   ├── position_encoding.py         # 2D 正弦位置编码
│   │   └── segmentor_v2.py              # DINOv3-Mask2Former 完整模型 (FPN + Transformer Decoder)
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── landslide_dataset.py         # 统一数据集类 (Bijie PNG + Moxizhen TIFF)
│   │   └── transforms.py               # 数据增强流水线
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py                   # 训练循环
│   │   └── evaluator.py                # 评测 (mIoU, F1, P/R)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                   # 指标计算
│       └── visualization.py             # 可视化工具
├── scripts/
│   ├── train.py                         # 训练入口脚本 (支持 DDP 多卡)
│   ├── evaluate.py                      # 评测入口脚本
│   ├── train_ablation.py                # 消融实验脚本
│   ├── train_baseline.py                # 基线对比实验脚本
│   ├── smoke_test.py                    # 全流程冒烟测试
│   ├── run_experiments.sh               # 一键运行全部实验
│   └── summarize_results.py             # 汇总实验结果
├── requirements.txt                     # Python 依赖
├── pretrained/                          # 预训练权重 (符号链接或路径)
│   └── README.md                        # 说明预训练权重位置
├── outputs/                             # 训练输出 (checkpoints, logs)
├── landslide_detr/                      # [归档] 旧DETR代码，仅供参考
├── landslide_/                          # 数据集根目录
│   ├── Bijie_landslide_dataset/         # Bijie数据 (源域)
│   │   └── Bijie-landslide-dataset/
│   │       ├── landslide/
│   │       │   ├── image/               # 633张 PNG RGB图像
│   │       │   ├── mask/                # 633张 PNG 二值掩膜
│   │       │   ├── dem/                 # [不使用] DEM数据
│   │       │   └── polygon_coordinate/  # [不使用] 多边形坐标
│   │       └── non-landslide/
│   │           ├── image/               # 2078张 PNG RGB图像
│   │           └── dem/                 # [不使用]
│   └── moxizhen/                        # Moxizhen v1数据 (目标域)
│       └── moxizhen/
│           ├── img/                     # 1635张 TIFF RGB图像 (UAV 0.2m)
│           ├── label/                   # 1634张 TIFF 标注 (滑坡区域)
│           └── mask/                    # [不使用]
├── dinov3-vitl16-pretrain-sat493m/      # DINOv2-ViT-L 卫星预训练权重
├── mask2former-swin-large-coco-panoptic/# Mask2Former预训练权重 (参考)
├── Mask2Former/                         # [参考] Mask2Former官方代码
└── dinov3/                              # [参考] DINOv2官方代码
```

## 数据集详情

### Bijie Landslide Dataset (源域/训练)
- **来源**: 卫星遥感 (SAT)
- **格式**: PNG
- **滑坡样本**: 633张 (image + mask 一一对应, 文件名如 js001.png, ny001.png 等)
- **非滑坡样本**: 2078张 (仅 image, 无 mask)
- **mask**: 二值掩膜, 0=背景, 255(或1)=滑坡
- **用途**: 训练集 + 验证集 (按比例划分, 如 8:2)

### Moxizhen v1 (目标域/跨域测试)
- **来源**: 无人机 (UAV), 0.2m分辨率
- **格式**: TIFF (GeoTIFF)
- **样本数**: 1635张 img, 1634张 label
- **label**: TIFF格式标注, 记录滑坡区域信息
- **用途**: 零样本跨域测试 (不参与训练)

## 模型架构
```
Input Image (RGB, 512x512)
    │
    ▼
DINOv3 Backbone (冻结, ViT-L/16, 1024维, RoPE + 4 register tokens)
    │ ── LoRA Adapters (可训练, q_proj/v_proj, rank=16)
    │
    ▼
Multi-scale Feature Extraction (层 5/11/17/23, 各 1024维)
    │
    ▼
FPN Pixel Decoder (官方 Mask2Former 移植, lateral+top-down, 256维)
    │ ── mask_features: (B, 256, H/16, W/16)
    │ ── multi_scale_features: 3层 FPN 特征
    │
    ▼
MultiScaleMaskedTransformerDecoder (官方 Mask2Former 核心)
    │ ── 9层 decoder (cross-attn + self-attn + FFN)
    │ ── Masked cross-attention (前一层 mask 指导下一层 attention)
    │ ── 多尺度轮询 (3层 FPN 特征循环 attend)
    │ ── 100 learnable queries (query_feat + query_embed)
    │ ── Level embedding (区分不同尺度)
    │ ── 每层中间预测 (auxiliary loss)
    │
    ▼
Output:
    ── pred_logits: (B, Q, K+1) 分类 logits
    ── pred_masks: (B, Q, H, W) mask predictions
    ── seg_logits: (B, K, H, W) 语义分割输出 (2类: 背景 + 滑坡)
```

### LoRA 配置
- 插入位置: DINOv3 attention 层的 `q_proj`, `v_proj` (注意: 不是 DINOv2 的 query/value)
- 秩 r: 默认 16 (可调)
- 缩放系数 alpha: 默认 32
- 可训练参数: LoRA 1,572,864 (0.52% backbone), 总可训练 9,658,627 (3.09% 全模型)

### 损失函数 (官方 Mask2Former SetCriterion)
- **Hungarian Matching**: 将 query predictions 与 ground truth 一对一匹配
- **分类损失**: Cross-Entropy (weight=5.0, no-object eos_coef=0.1)
- **Mask 损失**: Sigmoid BCE + Dice (基于 point sampling, 12544 点)
- **Auxiliary Loss**: 每层 decoder 都计算中间损失 (同权重)

## 实验方案

### 实验1: 跨平台泛化 (主实验)
- **训练**: Bijie (SAT) 全部滑坡数据
- **测试**: Moxizhen v1 (UAV)
- **目标**: 证明卫星数据训练的模型可在无人机图像上检测滑坡

### 消融实验
- LoRA vs 全量微调 vs 冻结backbone
- 不同 LoRA rank (4, 8, 16, 32)
- 不同数据增强策略
- 不同输入分辨率

### 对比方法
- U-Net
- DeepLabv3+
- YOLOv8 (seg)
- YOLOv10 (seg)

### 评价指标
- mIoU (mean Intersection over Union)
- F1 Score
- Precision
- Recall

## 开发阶段

### Phase 1: 基础设施 [x]
- [x] 清理项目文件夹
- [x] 创建项目计划文档
- [x] 创建新项目结构 (src/, configs/, scripts/)
- [x] 编写 requirements.txt
- [x] 实现统一数据集类
- [x] 实现数据增强流水线 (含 DINOv3 卫星归一化参数)

### Phase 2: 模型实现 [x]
- [x] DINOv3 backbone + LoRA 封装 (含 register tokens 处理)
- [x] Mask2Former 风格分割头 (FPN + Transformer Decoder)
- [x] 完整模型构建入口

### Phase 3: 训练与评测 [x]
- [x] 训练循环实现 (CE + Dice loss)
- [x] 评测管线 (mIoU, F1, P/R)
- [x] 可视化工具
- [x] CPU 小样本跑通全流程 (smoke_test.py ALL PASSED)

### Phase 4: 服务器部署与正式训练
- [ ] 服务器安装 transformers>=4.56.0 + peft
- [ ] scp 代码到服务器
- [ ] 使用 ViT-L 卫星预训练模型正式训练 (切换归一化为 sat)
- [ ] 跨域评测
- [ ] 对比实验与消融实验

## 环境配置
- **本地 Mac**: conda env `ad`, Python 3.x, CPU only
- **服务器**: GPU (CUDA), 正式训练
- 所有脚本支持 `--device cpu/cuda` 自动切换
