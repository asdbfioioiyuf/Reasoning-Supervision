"""
Two-Stream Video Transformer for Action Recognition
Multi-scale temporal reasoning with spatial-temporal attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TokenDropout(nn.Module):
    """Stochastic token dropout for regularization."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, T, N, D = x.shape
        mask = x.new_empty(B, T, N, 1).bernoulli_(1-self.p).div_(1-self.p)
        return x * mask


def get_1d_sincos(L, dim):
    """Generate 1D sinusoidal positional embeddings."""
    L = int(L)
    dim = int(dim)
    position = torch.arange(int(L), dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros((L, dim), dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def get_2d_sincos(h, w, dim):
    """Generate 2D sinusoidal positional embeddings."""
    assert dim % 4 == 0, "embed_dim must be divisible by 4"
    d4 = dim // 4

    y = torch.arange(h, dtype=torch.float32)[:, None]
    x = torch.arange(w, dtype=torch.float32)[:, None]

    omega = torch.exp(torch.arange(d4, dtype=torch.float32) * (-math.log(10000.0) / d4))

    wy = y * omega[None, :]
    wx = x * omega[None, :]

    pos_y = torch.cat([torch.sin(wy), torch.cos(wy)], dim=1)
    pos_x = torch.cat([torch.sin(wx), torch.cos(wx)], dim=1)

    pos = torch.cat([
        pos_y[:, None, :].expand(-1, w, -1),
        pos_x[None, :, :].expand(h, -1, -1)
    ], dim=2)

    return pos.permute(2, 0, 1).unsqueeze(0).contiguous()


class PatchEmbedding(nn.Module):
    """Patch embedding from pre-extracted features."""

    def __init__(self, feature_dim=2048, spatial_size=4, embed_dim=384):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.num_patches = spatial_size * spatial_size

        self.proj = nn.Linear(feature_dim, embed_dim)
        self.register_buffer("pos_embed", get_2d_sincos(spatial_size, spatial_size, embed_dim), persistent=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] - Pre-extracted features

        Returns:
            [B, T, N, embed_dim] - Patch tokens
        """
        B, T, C, H, W = x.shape
        assert C == self.feature_dim
        assert H == self.spatial_size and W == self.spatial_size

        x = x.view(B * T, C, H, W)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        pos_embed_flat = self.pos_embed.flatten(2).transpose(1, 2)
        x = x + pos_embed_flat

        x = self.norm(x)
        x = x.view(B, T, self.num_patches, -1)

        return x


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        y, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        return y, attn


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and MLP."""
    def __init__(self, embed_dim=64, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        x1 = self.ln1(x)
        y, attn = self.attn(x1)
        x = x + self.dropout(y)
        x2 = self.ln2(x)
        x2 = self.mlp(x2)
        x = x + self.dropout(x2)
        if return_attn:
            return x, attn
        return x, None


class SpatialTransformer(nn.Module):
    """Spatial transformer for processing frame-level features."""

    def __init__(self, feature_dim=2048, spatial_size=4, embed_dim=384,
                 num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(feature_dim=feature_dim, spatial_size=spatial_size, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.token_dropout = TokenDropout(dropout)

        self.grid_h = self.patch_embed.spatial_size
        self.grid_w = self.patch_embed.spatial_size

    def forward(self, x, return_attn=False):
        """
        Args:
            x: [B, T, C, H, W] - Pre-extracted features

        Returns:
            frame_features: [B, T, embed_dim]
            patch_tokens: [B, T, N, embed_dim]
            attn_spatial: [B*T, heads, 1+N, 1+N] (if requested)
        """
        B, T, C, H, W = x.shape

        patch_tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, T, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=2)
        tokens = self.token_dropout(tokens)

        BT = B * T
        N = patch_tokens.shape[2]
        tokens = tokens.view(BT, 1 + N, -1)

        attn_spatial = None
        for i, block in enumerate(self.blocks):
            get_attn = return_attn and i == len(self.blocks) - 1
            tokens, attn = block(tokens, return_attn=get_attn)
            if attn is not None:
                attn_spatial = attn

        tokens = self.ln(tokens)

        cls_tokens = tokens[:, 0, :]
        patch_tokens_out = tokens[:, 1:, :]

        frame_features = cls_tokens.view(B, T, -1)
        patch_tokens_out = patch_tokens_out.view(B, T, N, -1)

        return frame_features, patch_tokens_out, attn_spatial


class TemporalTransformer(nn.Module):
    """Temporal transformer for processing frame sequences."""

    def __init__(self, embed_dim=384, num_heads=8, mlp_ratio=4.0, dropout=0.1, max_T=64, num_layers=3):
        super().__init__()
        pe = get_1d_sincos(max_T + 1, embed_dim)
        self.register_buffer("pos_temporal", pe, persistent=False)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.clip_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.clip_cls, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x: [B, T_sub, D] - Frame features

        Returns:
            clip_cls: [B, D]
            frame_tokens: [B, T_sub, D]
            attn_maps: [B, heads, 1+T_sub, 1+T_sub]
        """
        B, T_sub, D = x.shape

        cls = self.clip_cls.expand(B, 1, -1)
        y = torch.cat([cls, x], dim=1)

        if T_sub + 1 > self.pos_temporal.shape[0]:
            extended_pe = get_1d_sincos(T_sub + 1, D).to(y.device)
            pos = extended_pe.unsqueeze(0)
        else:
            pos = self.pos_temporal[: T_sub + 1, :].unsqueeze(0).to(y.device)

        y = y + pos

        attn_maps = None
        for i, blk in enumerate(self.blocks):
            y, attn = blk(y, return_attn=(return_attn and (i == len(self.blocks) - 1)))
            if attn is not None:
                attn_maps = attn

        y = self.ln(y)
        clip_cls = y[:, 0, :]
        frame_tokens = y[:, 1:, :]
        return clip_cls, frame_tokens, attn_maps


class MultiScaleTemporalStage(nn.Module):
    """Multi-scale temporal reasoning stage."""

    def __init__(self, embed_dim=384, num_heads=8, mlp_ratio=4.0, dropout=0.1, max_T=64,
                 num_layers=3, num_classes=101, temperature=4.0):
        super().__init__()

        # Three temporal transformers for different scales
        self.immediate_vit = TemporalTransformer(embed_dim, num_heads, mlp_ratio, dropout, max_T, num_layers)
        self.context_vit = TemporalTransformer(embed_dim, num_heads, mlp_ratio, dropout, max_T, num_layers)
        self.narrative_vit = TemporalTransformer(embed_dim, num_heads, mlp_ratio, dropout, max_T, num_layers)

        # Cross-temporal reasoning
        self.causal_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_reasoning = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(2)
        ])

        # Multi-temporal fusion
        self.temporal_fusion = nn.Linear(embed_dim * 3, embed_dim)

        # Classification heads
        self.immediate_head = nn.Linear(embed_dim, num_classes)
        self.contextual_head = nn.Linear(embed_dim, num_classes)
        self.narrative_head = nn.Linear(embed_dim, num_classes)

        self.temperature = temperature

    def _adaptive_temporal_segmentation(self, T: int):
        """Adaptive temporal segmentation based on clip length."""
        if T <= 4:
            return 0, T, 0, T
        elif T <= 8:
            immediate_start = T // 2
            return immediate_start, T, 0, T
        elif T <= 16:
            immediate_start = 3 * T // 4
            context_start = T // 4
            context_end = 3 * T // 4
            return immediate_start, T, context_start, context_end
        else:
            immediate_length = max(4, T // 8)
            context_length = max(8, T // 4)

            immediate_start = T - immediate_length
            context_start = max(0, T // 2 - context_length // 2)
            context_end = min(T, context_start + context_length)

            return immediate_start, T, context_start, context_end

    def forward(self, frame_features: torch.Tensor, return_attn: bool = False):
        """
        Args:
            frame_features: [B, T, D] - Frame-level features

        Returns:
            Dictionary with logits, features, and attention maps
        """
        B, T, D = frame_features.shape

        # Adaptive temporal segmentation
        imm_start, imm_end, ctx_start, ctx_end = self._adaptive_temporal_segmentation(T)

        # Multi-scale temporal reasoning
        immediate_frames = frame_features[:, imm_start:imm_end, :]
        immediate_cls, _, immediate_attn = self.immediate_vit(immediate_frames, return_attn=return_attn)

        context_frames = frame_features[:, ctx_start:ctx_end, :]
        context_cls, _, context_attn = self.context_vit(context_frames, return_attn=return_attn)

        narrative_cls, _, narrative_attn = self.narrative_vit(frame_features, return_attn=return_attn)

        # Cross-temporal reasoning
        temporal_queries = torch.stack([immediate_cls, context_cls, narrative_cls], dim=1)

        temporal_attn = None
        for i, blk in enumerate(self.temporal_reasoning):
            temporal_queries, attn = blk(temporal_queries, return_attn=(return_attn and i == len(self.temporal_reasoning)-1))
            if attn is not None:
                temporal_attn = attn

        # Causal reasoning
        _, causal_attn = self.causal_attention(
            narrative_cls.unsqueeze(1),
            frame_features,
            frame_features
        )

        # Extract enhanced representations
        enhanced_immediate = temporal_queries[:, 0]
        enhanced_context = temporal_queries[:, 1]
        enhanced_narrative = temporal_queries[:, 2]

        # Fusion
        fused_temporal = self.temporal_fusion(torch.cat([
            enhanced_immediate,
            enhanced_context,
            enhanced_narrative
        ], dim=-1))

        # Multi-level predictions
        immediate_logits = self.immediate_head(enhanced_immediate)
        contextual_logits = self.contextual_head(enhanced_context)
        narrative_logits = self.narrative_head(enhanced_narrative)

        # Temperature-scaled probabilities
        immediate_probs = F.softmax(immediate_logits / self.temperature, dim=-1)
        contextual_probs = F.softmax(contextual_logits / self.temperature, dim=-1)
        narrative_probs = F.softmax(narrative_logits / self.temperature, dim=-1)

        # Temporal consistency loss
        temporal_consistency_loss = self._compute_temporal_consistency(
            immediate_probs, contextual_probs, narrative_probs
        )

        # Adaptive final prediction
        if T <= 4:
            final_logits = (immediate_logits * 2.0 + contextual_logits + narrative_logits) / 4.0
        elif T <= 16:
            final_logits = (immediate_logits + contextual_logits * 1.5 + narrative_logits * 2.0) / 4.5
        else:
            final_logits = (immediate_logits + contextual_logits + narrative_logits * 3.0) / 5.0

        return {
            "logits": final_logits,
            "immediate_logits": immediate_logits,
            "contextual_logits": contextual_logits,
            "narrative_logits": narrative_logits,
            "temporal_consistency_loss": temporal_consistency_loss,
            "enhanced_features": fused_temporal,
            "immediate_cls": enhanced_immediate,
            "contextual_cls": enhanced_context,
            "narrative_cls": enhanced_narrative,
            "immediate_attn": immediate_attn,
            "contextual_attn": context_attn,
            "narrative_attn": narrative_attn,
            "causal_attn": causal_attn,
            "temporal_attn": temporal_attn,
        }

    def _compute_temporal_consistency(self, immediate_probs, contextual_probs, narrative_probs):
        """Compute consistency loss between different temporal scales."""
        kl_immediate_context = F.kl_div(torch.log(immediate_probs + 1e-8), contextual_probs, reduction='batchmean')
        kl_context_immediate = F.kl_div(torch.log(contextual_probs + 1e-8), immediate_probs, reduction='batchmean')

        kl_context_narrative = F.kl_div(torch.log(contextual_probs + 1e-8), narrative_probs, reduction='batchmean')
        kl_narrative_context = F.kl_div(torch.log(narrative_probs + 1e-8), contextual_probs, reduction='batchmean')

        kl_immediate_narrative = F.kl_div(torch.log(immediate_probs + 1e-8), narrative_probs, reduction='batchmean')
        kl_narrative_immediate = F.kl_div(torch.log(narrative_probs + 1e-8), immediate_probs, reduction='batchmean')

        total_consistency_loss = (
            (kl_immediate_context + kl_context_immediate) / 2.0 +
            (kl_context_narrative + kl_narrative_context) / 2.0 +
            (kl_immediate_narrative + kl_narrative_immediate) / 2.0
        ) / 3.0

        return total_consistency_loss


class VideoActionRecognitionModel(nn.Module):
    """Two-stream video transformer for action recognition."""

    def __init__(
        self,
        feature_dim: int = 2048,
        spatial_size: int = 4,
        embed_dim: int = 384,
        spatial_layers: int = 4,
        spatial_heads: int = 6,
        temporal_layers: int = 3,
        temporal_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_T: int = 64,
        num_classes: int = 101,
        temperature: float = 4.0
    ):
        super().__init__()

        # Stage 1: Spatial processing
        self.stage1 = SpatialTransformer(
            feature_dim=feature_dim,
            spatial_size=spatial_size,
            embed_dim=embed_dim,
            num_layers=spatial_layers,
            num_heads=spatial_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Stage 2: Multi-scale temporal reasoning
        self.stage2 = MultiScaleTemporalStage(
            embed_dim=embed_dim,
            num_heads=temporal_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_T=max_T,
            num_layers=temporal_layers,
            num_classes=num_classes,
            temperature=temperature
        )

        self.grid_h = self.stage1.grid_h
        self.grid_w = self.stage1.grid_w

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x: [B, T, C, H, W] - Pre-extracted features

        Returns:
            Dictionary with predictions, features, and attention maps
        """
        if x.ndim == 4:
            x = x.unsqueeze(0)
        assert x.ndim == 5, "Input must be [B,T,C,H,W] or [T,C,H,W]."

        # Stage 1: Spatial processing
        frame_features, patch_tokens, attn_spatial = self.stage1(x, return_attn=return_attn)

        # Stage 2: Multi-scale temporal reasoning
        stage2_output = self.stage2(frame_features, return_attn=return_attn)

        return {
            "logits": stage2_output["logits"],
            "immediate_logits": stage2_output["immediate_logits"],
            "contextual_logits": stage2_output["contextual_logits"],
            "narrative_logits": stage2_output["narrative_logits"],
            "temporal_consistency_loss": stage2_output["temporal_consistency_loss"],
            "frame_features": frame_features,
            "enhanced_features": stage2_output["enhanced_features"],
            "immediate_cls": stage2_output["immediate_cls"],
            "contextual_cls": stage2_output["contextual_cls"],
            "narrative_cls": stage2_output["narrative_cls"],
            "patch_tokens": patch_tokens,
            "attn_spatial": attn_spatial,
            "immediate_attn": stage2_output["immediate_attn"],
            "contextual_attn": stage2_output["contextual_attn"],
            "narrative_attn": stage2_output["narrative_attn"],
            "causal_attn": stage2_output["causal_attn"],
            "temporal_attn": stage2_output["temporal_attn"],
        }
