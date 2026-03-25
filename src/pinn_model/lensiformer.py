"""
Consistency-Regularized Lensiformer — Full Architecture.

Combines:
1. Physics Encoder (Inverse Lens Layer)
2. Shifted Patch Tokenization (SPT)
3. Learnable Gating Fusion (original + reconstructed source)
4. Hybrid CNN → ViT with Locality Self-Attention (LSA)
5. MC Dropout for uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import config
from .lens_layer import InverseLensLayer


# ─── Shifted Patch Tokenization (SPT) ──────────────────────────────────────

class ShiftedPatchTokenization(nn.Module):
    """
    SPT from ViTSD paper (Lee et al., 2021).
    
    Shifts image in 4 diagonal directions, concatenates with original → 5× channels.
    Then extracts patches and projects to token embeddings.
    Captures local spatial relationships that standard ViT misses.
    """
    
    def __init__(self, in_channels, patch_size, embed_dim, shift_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.shift_size = shift_size
        
        # 5 shifts: original + 4 diagonals
        total_channels = in_channels * 5
        
        # Patch embedding: (5*C, P, P) patches → embed_dim tokens
        self.projection = nn.Conv2d(
            total_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image
        Returns:
            tokens: (B, num_patches, embed_dim)
        """
        # Diagonal shifts
        shifts = [
            x,  # Original
            torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3)),
            torch.roll(x, shifts=(-self.shift_size, self.shift_size), dims=(2, 3)),
            torch.roll(x, shifts=(self.shift_size, -self.shift_size), dims=(2, 3)),
            torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)),
        ]
        
        x_shifted = torch.cat(shifts, dim=1)  # (B, 5*C, H, W)
        
        # Patch embedding
        tokens = self.projection(x_shifted)   # (B, embed_dim, H/P, W/P)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        tokens = self.norm(tokens)
        
        return tokens


# ─── Locality Self-Attention (LSA) ─────────────────────────────────────────

class LocalitySelfAttention(nn.Module):
    """
    LSA from ViTSD paper.
    
    Standard multi-head self-attention with:
    1. Learnable temperature for attention scaling
    2. Diagonal masking to prevent self-token attention
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature — initialize to 1/√head_dim (standard attention scaling)
        # Starting at 1.0 was too small, causing uniform attention and gradient collapse
        self.temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1) / math.sqrt(self.head_dim)
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        # Diagonal mask: prevent attending to self
        mask = torch.eye(N, device=x.device, dtype=torch.bool)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ─── Transformer Block ─────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Transformer block with LSA + MLP + residual connections."""
    
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LocalitySelfAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─── CNN Backbone (for hybrid approach) ────────────────────────────────────

class CNNBackbone(nn.Module):
    """
    Lightweight CNN to extract low-level features (arcs, edges, rings).
    Used before ViT in the hybrid approach.
    """
    
    def __init__(self, in_channels=1, out_features=None):
        super().__init__()
        out_features = out_features or config.CNN_FEATURES
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 150→75
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_features, 3, stride=2, padding=1),  # 75→38
            nn.BatchNorm2d(out_features),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        return self.features(x)


# ─── Learnable Gating Fusion ───────────────────────────────────────────────

class LearnableGatingFusion(nn.Module):
    """
    Fuses source and original features using learnable gating.
    
    gate = σ(W · [source_feat, original_feat])
    fused = gate * source_feat + (1−gate) * original_feat
    
    Forces interaction between branches instead of simple concat.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, source_tokens, original_tokens):
        """
        Args:
            source_tokens: (B, N, D) from reconstructed source
            original_tokens: (B, N, D) from original image
        Returns:
            fused: (B, N, D) gated fusion
        """
        combined = torch.cat([source_tokens, original_tokens], dim=-1)  # (B, N, 2D)
        g = self.gate(combined)  # (B, N, D)
        fused = g * source_tokens + (1.0 - g) * original_tokens
        return fused


# ─── Full Lensiformer ──────────────────────────────────────────────────────

class Lensiformer(nn.Module):
    """
    Consistency-Regularized Lensiformer.
    
    Physics Encoder → Dual SPT → Learnable Fusion → Hybrid CNN→ViT → Classifier
    """
    
    def __init__(
        self,
        num_classes=None,
        patch_size=None,
        vit_dim=None,
        vit_depth=None,
        vit_heads=None,
        vit_mlp_dim=None,
        dropout=None,
        mc_dropout=None
    ):
        super().__init__()
        
        num_classes = num_classes or config.NUM_CLASSES
        patch_size = patch_size or config.PATCH_SIZE
        vit_dim = vit_dim or config.VIT_DIM
        vit_depth = vit_depth or config.VIT_DEPTH
        vit_heads = vit_heads or config.VIT_HEADS
        vit_mlp_dim = vit_mlp_dim or config.VIT_MLP_DIM
        dropout = dropout if dropout is not None else config.VIT_DROPOUT
        mc_dropout = mc_dropout if mc_dropout is not None else config.MC_DROPOUT_RATE
        
        # 1. Physics Encoder (Inverse Lens Layer)
        self.inverse_lens = InverseLensLayer()
        
        # 2. SPT for both source and original
        self.spt_source = ShiftedPatchTokenization(1, patch_size, vit_dim)
        self.spt_original = ShiftedPatchTokenization(1, patch_size, vit_dim)
        
        # 3. Learnable Gating Fusion
        self.fusion = LearnableGatingFusion(vit_dim)
        
        # 4. CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim) * 0.02)
        
        # 5. Positional embedding
        num_patches = (config.IMAGE_SIZE // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, vit_dim) * 0.02
        )
        
        # 6. Transformer blocks
        self.transformer = nn.Sequential(*[
            TransformerBlock(vit_dim, vit_heads, vit_mlp_dim, dropout)
            for _ in range(vit_depth)
        ])
        
        # 7. Classification head with MC Dropout
        self.norm = nn.LayerNorm(vit_dim)
        self.mc_dropout = nn.Dropout(mc_dropout)
        self.head = nn.Linear(vit_dim, num_classes)
    
    def forward(self, image, return_physics=False):
        """
        Forward pass.
        
        Args:
            image: (B, 1, H, W) lensed image in [0, 1]
            return_physics: If True, return physics_data for losses/visualization
            
        Returns:
            logits: (B, num_classes)
            physics_data: (optional) dict with k, alpha, source, etc.
        """
        B = image.shape[0]
        
        # Physics encoder: reconstruct source
        source, physics_data = self.inverse_lens(image)
        
        # SPT: tokenize both source and original
        source_tokens = self.spt_source(source)       # (B, N, D)
        original_tokens = self.spt_original(image)     # (B, N, D)
        
        # Learnable gating fusion
        fused_tokens = self.fusion(source_tokens, original_tokens)  # (B, N, D)
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, fused_tokens], dim=1)  # (B, N+1, D)
        
        # Add positional embedding
        tokens = tokens + self.pos_embed
        
        # Transformer
        tokens = self.transformer(tokens)
        
        # Classification: CLS token
        cls_output = self.norm(tokens[:, 0])
        cls_output = self.mc_dropout(cls_output)
        logits = self.head(cls_output)
        
        if return_physics:
            return logits, physics_data
        return logits
    
    @torch.no_grad()
    def predict_with_uncertainty(self, image, n_samples=None):
        """
        MC Dropout prediction with uncertainty estimation.
        
        Args:
            image: (B, 1, H, W)
            n_samples: Number of MC forward passes
            
        Returns:
            mean_probs: (B, num_classes)  mean prediction
            std_probs: (B, num_classes)   uncertainty
        """
        n_samples = n_samples or config.MC_SAMPLES
        
        self.train()  # Keep dropout active
        
        all_probs = []
        for _ in range(n_samples):
            logits = self.forward(image)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        all_probs = torch.stack(all_probs, dim=0)  # (N, B, C)
        
        self.eval()
        
        return all_probs.mean(dim=0), all_probs.std(dim=0)
