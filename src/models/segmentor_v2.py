"""
DINOv3-LoRA-Mask2Former 滑坡语义分割模型 (完整版)

基于 Meta 官方 Mask2Former 源码移植, 去除 detectron2 依赖.
核心组件:
  - FPN Pixel Decoder: 多尺度特征金字塔
  - MultiScaleMaskedTransformerDecoder: masked cross-attention + 多尺度轮询 + auxiliary predictions
  - SetCriterion + HungarianMatcher: query-level 匹配损失

Reference: https://github.com/facebookresearch/Mask2Former
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine


# ============================================================
#  Transformer Decoder 基础层 (from official Mask2Former)
# ============================================================

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):
        if self.normalize_before:
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
        else:
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
        return tgt


class MLP(nn.Module):
    """Multi-layer perceptron (from official Mask2Former)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ============================================================
#  FPN Pixel Decoder (去 detectron2 依赖)
# ============================================================

class FPNPixelDecoder(nn.Module):
    """
    FPN Pixel Decoder — 从官方 BasePixelDecoder 简化移植.
    将 backbone 多尺度特征转为 FPN 金字塔 + mask_features.

    输入: list of (B, C_i, H_i, W_i) — backbone 多尺度特征 (从高分辨率到低分辨率)
    输出:
      - mask_features: (B, mask_dim, H_0, W_0) 最高分辨率特征
      - multi_scale_features: list of 3 个 (B, conv_dim, H_i, W_i) 用于 Transformer Decoder
    """

    def __init__(self, in_channels_list, conv_dim=256, mask_dim=256, num_feature_levels=3):
        """
        Args:
            in_channels_list: 每个尺度的输入通道数 (从高分辨率到低分辨率)
            conv_dim: FPN 中间通道数
            mask_dim: mask_features 输出通道数
            num_feature_levels: 输出给 Transformer Decoder 的尺度数 (默认3, 与官方一致)
        """
        super().__init__()
        self.num_feature_levels = num_feature_levels

        # Lateral 1x1 convs (除了最后一层)
        self.lateral_convs = nn.ModuleList()
        # Output 3x3 convs
        self.output_convs = nn.ModuleList()

        num_in = len(in_channels_list)
        for idx in range(num_in):
            in_ch = in_channels_list[idx]
            if idx == num_in - 1:
                # 最低分辨率层: 没有 lateral, 直接 3x3
                self.lateral_convs.append(None)
                output_conv = nn.Sequential(
                    nn.Conv2d(in_ch, conv_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(32, conv_dim),
                    nn.ReLU(inplace=True),
                )
            else:
                lateral_conv = nn.Sequential(
                    nn.Conv2d(in_ch, conv_dim, 1, bias=False),
                    nn.GroupNorm(32, conv_dim),
                )
                self.lateral_convs.append(lateral_conv)
                output_conv = nn.Sequential(
                    nn.Conv2d(conv_dim, conv_dim, 3, padding=1, bias=False),
                    nn.GroupNorm(32, conv_dim),
                    nn.ReLU(inplace=True),
                )
            self.output_convs.append(output_conv)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # mask_features: 最终的高分辨率特征投影
        self.mask_features = nn.Conv2d(conv_dim, mask_dim, 3, padding=1)
        nn.init.xavier_uniform_(self.mask_features.weight)
        nn.init.constant_(self.mask_features.bias, 0)

    def forward(self, features):
        """
        Args:
            features: list of (B, C_i, H_i, W_i), 从高分辨率到低分辨率

        Returns:
            mask_features: (B, mask_dim, H_0, W_0)
            multi_scale_features: list of (B, conv_dim, H_i, W_i), 从低分辨率到高分辨率
                                  (与官方一致, Transformer Decoder 期望此顺序)
        """
        num_in = len(features)

        # 反转为 top-down 顺序 (低分辨率 -> 高分辨率)
        multi_scale_features = []
        y = None
        for idx in range(num_in - 1, -1, -1):
            x = features[idx]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]

            if lateral_conv is None:
                # 最低分辨率层
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)

            if len(multi_scale_features) < self.num_feature_levels:
                multi_scale_features.append(y)

        # mask_features 是最高分辨率的
        mask_features = self.mask_features(y)

        return mask_features, multi_scale_features


# ============================================================
#  Multi-Scale Masked Transformer Decoder (官方 Mask2Former 核心)
# ============================================================

class MultiScaleMaskedTransformerDecoder(nn.Module):
    """
    Mask2Former 的核心 Transformer Decoder.
    从官方源码移植, 去除 detectron2 依赖.

    核心创新:
      1. Masked cross-attention: 用前一层的 mask prediction 生成 attention mask
      2. 多尺度轮询: decoder 层轮流 attend 不同尺度的特征
      3. Auxiliary loss: 每层 decoder 都输出中间预测
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
    ):
        """
        Args:
            in_channels: pixel decoder 输出的通道数 (conv_dim)
            num_classes: 类别数 (不含 no-object)
            hidden_dim: Transformer 特征维度
            num_queries: query 数量
            nheads: attention head 数量
            dim_feedforward: FFN 中间维度
            dec_layers: decoder 层数 (注意: 实际层数 = dec_layers - 1, 因为第0层是初始预测)
            pre_norm: 是否使用 pre-LayerNorm
            mask_dim: mask feature 维度
            enforce_input_project: 是否强制添加 input projection
        """
        super().__init__()

        # Positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Transformer decoder layers
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm)
            )
            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm)
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # Learnable query features (content) and query positional embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Multi-scale: always use 3 scales (与官方一致)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # Input projection (if needed)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, multi_scale_features, mask_features):
        """
        Args:
            multi_scale_features: list of 3 个 (B, C, H_i, W_i) 多尺度特征 (低分辨率到高分辨率)
            mask_features: (B, mask_dim, H, W) 最高分辨率特征

        Returns:
            dict with:
                "pred_logits": (B, Q, K+1) 最终分类 logits
                "pred_masks": (B, Q, H, W) 最终 mask predictions
                "aux_outputs": list of dicts, 每层 decoder 的中间预测
        """
        assert len(multi_scale_features) == self.num_feature_levels

        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            pos.append(self.pe_layer(multi_scale_features[i], None).flatten(2))  # (B, C, H*W)
            src.append(
                self.input_proj[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )  # (B, C, H*W)

            # Flatten NxCxHxW to HWxNxC (PyTorch MultiheadAttention expects seq_len first)
            pos[-1] = pos[-1].permute(2, 0, 1)   # (H*W, B, C)
            src[-1] = src[-1].permute(2, 0, 1)    # (H*W, B, C)

        _, bs, _ = src[0].shape

        # Query initialization: QxBxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q, B, C)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)        # (Q, B, C)

        predictions_class = []
        predictions_mask = []

        # Initial prediction (layer 0)
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # Masked cross-attention: 核心创新!
            # 将全部为 True 的行设为 False, 避免 NaN
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
            )

            output = self.transformer_ffn_layers[i](output)

            # 每层都做预测 (auxiliary loss)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],   # (B, Q, K+1)
            "pred_masks": predictions_mask[-1],      # (B, Q, H, W)
            "aux_outputs": [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(predictions_class[:-1], predictions_mask[:-1])
            ],
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """
        从 decoder output 生成分类和 mask 预测, 以及下一层的 attention mask.

        Args:
            output: (Q, B, C) decoder output
            mask_features: (B, C, H, W) pixel decoder 的 mask features
            attn_mask_target_size: (H_target, W_target) attention mask 的目标尺寸

        Returns:
            outputs_class: (B, Q, K+1)
            outputs_mask: (B, Q, H, W)
            attn_mask: (B*num_heads, Q, H_target*W_target) bool mask for next layer
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (B, Q, C)

        outputs_class = self.class_embed(decoder_output)   # (B, Q, K+1)
        mask_embed = self.mask_embed(decoder_output)        # (B, Q, mask_dim)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)  # (B, Q, H, W)

        # 生成 attention mask: 用当前 mask prediction 指导下一层的 cross-attention
        # [B, Q, H, W] -> interpolate -> [B, Q, H_t*W_t] -> [B, h, Q, H_t*W_t] -> [B*h, Q, H_t*W_t]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False
        )
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask


# ============================================================
#  完整模型: DINOv3 Backbone + FPN + Mask2Former Decoder
# ============================================================

class DINOv3Mask2Former(nn.Module):
    """
    DINOv3-LoRA-Mask2Former 完整分割模型.

    架构:
      DINOv3 (冻结+LoRA) → 多尺度特征 → FPN Pixel Decoder → Mask2Former Transformer Decoder
      → per-query class + mask predictions → 合并为语义分割输出
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        hidden_dim=256,
        num_queries=100,
        num_decoder_layers=9,
        nheads=8,
        dim_feedforward=2048,
        mask_dim=256,
        pre_norm=False,
    ):
        """
        Args:
            backbone: DINOv3Backbone 实例
            num_classes: 分割类别数 (不含 no-object)
            hidden_dim: Transformer decoder 隐藏维度
            num_queries: object query 数量
            num_decoder_layers: Transformer decoder 层数
            nheads: attention head 数量
            dim_feedforward: FFN 中间维度
            mask_dim: mask feature 维度
            pre_norm: 是否使用 pre-LayerNorm
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Pixel Decoder (FPN)
        in_channels = backbone.get_out_channels()  # list of channel dims per scale
        self.pixel_decoder = FPNPixelDecoder(
            in_channels_list=in_channels,
            conv_dim=hidden_dim,
            mask_dim=mask_dim,
        )

        # Mask2Former Transformer Decoder
        self.predictor = MultiScaleMaskedTransformerDecoder(
            in_channels=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=num_decoder_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
        )

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, H, W) 输入图像

        Returns:
            dict with:
                "pred_logits": (B, Q, K+1) 分类 logits
                "pred_masks": (B, Q, H, W) mask predictions (上采样到原始分辨率)
                "aux_outputs": list of intermediate predictions
                "seg_logits": (B, K, H, W) 合并后的语义分割 logits (用于评测)
        """
        B, _, H, W = pixel_values.shape

        # 1. Backbone: 提取多尺度特征
        features = self.backbone(pixel_values)  # list of (B, C, H_i, W_i)

        # 2. Pixel Decoder (FPN)
        mask_features, multi_scale_features = self.pixel_decoder(features)

        # 3. Mask2Former Transformer Decoder
        outputs = self.predictor(multi_scale_features, mask_features)

        # 4. 上采样 mask predictions 到原始分辨率
        outputs["pred_masks"] = F.interpolate(
            outputs["pred_masks"], size=(H, W), mode="bilinear", align_corners=False
        )
        for aux in outputs["aux_outputs"]:
            aux["pred_masks"] = F.interpolate(
                aux["pred_masks"], size=(H, W), mode="bilinear", align_corners=False
            )

        # 5. 合并为语义分割输出 (用于评测和简单推理)
        outputs["seg_logits"] = self._merge_semantic(outputs, H, W)

        return outputs

    def _merge_semantic(self, outputs, H, W):
        """
        将 query-level predictions 合并为 per-pixel 语义分割.
        使用 argmax query assignment (与官方 Mask2Former 推理一致).

        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        pred_logits = outputs["pred_logits"]   # (B, Q, K+1)
        pred_masks = outputs["pred_masks"]     # (B, Q, H, W)

        # 去掉 no-object 类, 取前 K 个类的概率
        mask_cls = F.softmax(pred_logits, dim=-1)[..., :-1]  # (B, Q, K)
        mask_pred = pred_masks.sigmoid()  # (B, Q, H, W)

        # 语义分割: 对每个像素, 加权合并所有 query 的 mask
        # seg = sum_q (class_prob_q * mask_prob_q)
        # (B, Q, K) x (B, Q, H, W) -> (B, K, H, W)
        seg_logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)

        return seg_logits

    def get_trainable_params(self):
        """返回需要训练的参数 (LoRA + decoder 全部)"""
        params = []
        # LoRA 参数 (backbone 中 requires_grad=True 的部分)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params.append({"params": param, "name": f"backbone.{name}"})

        # Pixel Decoder 参数
        for name, param in self.pixel_decoder.named_parameters():
            params.append({"params": param, "name": f"pixel_decoder.{name}"})

        # Transformer Decoder 参数
        for name, param in self.predictor.named_parameters():
            params.append({"params": param, "name": f"predictor.{name}"})

        return params
