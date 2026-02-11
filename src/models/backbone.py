"""
DINOv3 Backbone + LoRA 适配器
- 冻结 DINOv3 主干
- 在 attention 的 Q, V 投影上插入 LoRA
- 提取多尺度特征供 Mask2Former 使用

DINOv3 特点 (相比 DINOv2):
- 使用 RoPE 位置编码
- 有 4 个 register tokens
- token 序列: [CLS] + [4 registers] + [h*w patches]
- 需要 transformers >= 4.56.0
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DINOv3Backbone(nn.Module):
    """
    DINOv3 特征提取器 + LoRA
    
    支持两种加载方式:
    1. HuggingFace 模型名 (如 "facebook/dinov3-vits16-pretrain-lvd1689m") -> 自动下载
    2. 本地路径 (如 "./dinov3-vitl16-pretrain-sat493m") -> 从本地加载
    """

    def __init__(
        self,
        model_name_or_path="facebook/dinov3-vits16-pretrain-lvd1689m",
        freeze=True,
        out_indices=None,
        lora_config=None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.freeze = freeze

        # 加载 DINOv3
        print(f"[DINOv3Backbone] Loading model: {model_name_or_path}")
        self.dinov3 = AutoModel.from_pretrained(model_name_or_path)
        self.config = self.dinov3.config

        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.num_layers = self.config.num_hidden_layers
        # DINOv3 有 register tokens (默认4个)
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 4)
        # 需要跳过的前缀 token 数: 1 (CLS) + num_register_tokens
        self.num_prefix_tokens = 1 + self.num_register_tokens

        # 多尺度特征提取层索引
        if out_indices is None:
            # 默认: 均匀取4层
            step = self.num_layers // 4
            self.out_indices = [step - 1, 2 * step - 1, 3 * step - 1, self.num_layers - 1]
        else:
            self.out_indices = out_indices
        # 确保索引不超出范围
        self.out_indices = [min(i, self.num_layers - 1) for i in self.out_indices]

        print(f"[DINOv3Backbone] hidden_size={self.hidden_size}, "
              f"patch_size={self.patch_size}, num_layers={self.num_layers}, "
              f"register_tokens={self.num_register_tokens}, "
              f"out_indices={self.out_indices}")

        # 冻结 backbone
        if freeze:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            print("[DINOv3Backbone] Backbone frozen")

        # 应用 LoRA
        if lora_config and lora_config.get("enabled", False):
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config):
        """使用 peft 库应用 LoRA 到 DINOv3"""
        try:
            from peft import LoraConfig, get_peft_model

            rank = lora_config.get("rank", 16)
            alpha = lora_config.get("alpha", 32)
            dropout = lora_config.get("dropout", 0.1)
            target_modules = lora_config.get("target_modules", ["q_proj", "v_proj"])

            # lora_layers: 只在指定的层插入 LoRA (0-indexed)
            # 例如 [20,21,22,23] 表示只在最后4层插入
            # 不配置则全部层都插入
            lora_layers = lora_config.get("lora_layers", None)

            if lora_layers is not None:
                # 用正则精确匹配指定层: layer.(20|21|22|23).attention.(q_proj|v_proj)
                layer_pattern = "|".join(str(l) for l in lora_layers)
                module_pattern = "|".join(target_modules)
                regex_target = rf"layer\.({layer_pattern})\.attention\.({module_pattern})"
                print(f"[DINOv3Backbone] LoRA only on layers: {lora_layers}")
                actual_targets = regex_target
            else:
                actual_targets = target_modules

            peft_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=actual_targets,
                bias="none",
            )
            self.dinov3 = get_peft_model(self.dinov3, peft_config)

            # 统计可训练参数
            trainable = sum(p.numel() for p in self.dinov3.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.dinov3.parameters())
            print(f"[DINOv3Backbone] LoRA applied: rank={rank}, alpha={alpha}")
            print(f"[DINOv3Backbone] Trainable params: {trainable:,} / {total:,} "
                  f"({100 * trainable / total:.2f}%)")

        except ImportError:
            print("[DINOv3Backbone] WARNING: peft not installed, LoRA disabled")
            print("[DINOv3Backbone] Install with: pip install peft")

    def forward(self, pixel_values):
        """
        前向传播, 提取多尺度特征
        
        Args:
            pixel_values: (B, 3, H, W) 输入图像 tensor
        
        Returns:
            features: list of (B, C, h, w) 多尺度特征图
                      len = len(out_indices), C = hidden_size
        """
        # 获取所有隐藏层输出
        outputs = self.dinov3(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # tuple of (B, N, C)
        # DINOv3: N = 1(CLS) + num_register_tokens + h*w(patches)

        # 计算空间尺寸
        B = pixel_values.shape[0]
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        h = H // self.patch_size
        w = W // self.patch_size

        # 从指定层提取特征, 去掉 CLS + register tokens, reshape为2D特征图
        features = []
        for idx in self.out_indices:
            # hidden_states[0] 是 embedding 输出, hidden_states[1] 是第1层输出
            # 所以 layer i 的输出在 hidden_states[i+1]
            layer_idx = idx + 1
            if layer_idx < len(hidden_states):
                feat = hidden_states[layer_idx]                  # (B, 1+reg+h*w, C)
                feat = feat[:, self.num_prefix_tokens:, :]       # 去掉 CLS + registers -> (B, h*w, C)
                feat = feat.permute(0, 2, 1)                     # (B, C, h*w)
                feat = feat.reshape(B, -1, h, w)                 # (B, C, h, w)
                features.append(feat)

        return features

    def get_out_channels(self):
        """返回每个尺度的输出通道数 (都是 hidden_size)"""
        return [self.hidden_size] * len(self.out_indices)
