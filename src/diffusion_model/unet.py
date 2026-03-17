"""
U-Net architecture for DDPM noise prediction.

Features:
- Sinusoidal timestep embeddings
- Residual blocks with GroupNorm + SiLU
- Self-attention at specified resolutions
- Skip connections between encoder and decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# ─── Sinusoidal Position Embedding ──────────────────────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al., 2017)."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ─── Building Blocks ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU activation, and time embedding injection.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection (match channels if needed)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_proj(t_emb)[:, :, None, None]
        h = h + t
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with residual connection."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)    # (B, C, H, W)
        
        return x + residual


class Downsample(nn.Module):
    """Spatial downsampling with strided convolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling with interpolation + convolution."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ─── U-Net ──────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for noise prediction in DDPM.
    
    Architecture:
        Input (B, 1, 128, 128) + timestep →
        Encoder (downsample 4x) → Bottleneck → Decoder (upsample 4x) →
        Output (B, 1, 128, 128)
    
    Args:
        in_channels: Input image channels (1 for grayscale)
        base_channels: Base channel count (doubled at each level)
        channel_mults: Channel multipliers for each level
        attention_resolutions: Spatial sizes where attention is applied
        num_res_blocks: Number of residual blocks per level
        time_emb_dim: Dimension of timestep embedding
        dropout: Dropout rate
        use_attention: Whether to use self-attention (for ablation)
    """
    
    def __init__(
        self,
        in_channels=None,
        base_channels=None,
        channel_mults=None,
        attention_resolutions=None,
        num_res_blocks=None,
        time_emb_dim=None,
        dropout=None,
        use_attention=True
    ):
        super().__init__()
        
        in_channels = in_channels or config.CHANNELS
        base_channels = base_channels or config.BASE_CHANNELS
        channel_mults = channel_mults or config.CHANNEL_MULTS
        attention_resolutions = attention_resolutions or config.ATTENTION_RESOLUTIONS
        num_res_blocks = num_res_blocks or config.NUM_RES_BLOCKS
        time_emb_dim = time_emb_dim or config.TIME_EMB_DIM
        dropout = dropout if dropout is not None else config.DROPOUT
        
        self.use_attention = use_attention
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # ─── Encoder ────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        channels = [base_channels]
        current_channels = base_channels
        current_res = config.IMAGE_SIZE
        
        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualBlock(current_channels, out_channels, time_emb_dim, dropout)
                )
                current_channels = out_channels
                
                # Add attention at specified resolutions
                if use_attention and current_res in attention_resolutions:
                    level_blocks.append(SelfAttention(current_channels))
            
            self.encoder_blocks.append(level_blocks)
            channels.append(current_channels)
            
            # Downsample (except at last level)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(current_channels))
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())
        
        # ─── Bottleneck ─────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_emb_dim, dropout),
            SelfAttention(current_channels) if use_attention else nn.Identity(),
            ResidualBlock(current_channels, current_channels, time_emb_dim, dropout),
        ])
        
        # ─── Decoder ────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # Concatenate skip connection
                skip_channels = channels.pop() if i == 0 else 0
                block_in = current_channels + skip_channels if i == 0 else current_channels
                
                level_blocks.append(
                    ResidualBlock(block_in, out_channels, time_emb_dim, dropout)
                )
                current_channels = out_channels
                
                # Add attention at specified resolutions
                if use_attention and current_res in attention_resolutions:
                    level_blocks.append(SelfAttention(current_channels))
            
            self.decoder_blocks.append(level_blocks)
            
            # Upsample (except at first decoder level)
            if level > 0:
                self.upsamplers.append(Upsample(current_channels))
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())
        
        # Final output
        self.final = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: Noisy image (B, C, H, W)
            t: Timestep (B,) integer tensor
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = [x]
        for level_blocks, downsampler in zip(self.encoder_blocks, self.downsamplers):
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    x = block(x, t_emb)
                else:
                    x = block(x)
            skips.append(x)
            x = downsampler(x)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, t_emb)
            elif isinstance(block, nn.Identity):
                pass
            else:
                x = block(x)
        
        # Decoder with skip connections
        for level_blocks, upsampler in zip(self.decoder_blocks, self.upsamplers):
            first_block = True
            for block in level_blocks:
                if isinstance(block, ResidualBlock):
                    if first_block:
                        skip = skips.pop()
                        x = torch.cat([x, skip], dim=1)
                        first_block = False
                    x = block(x, t_emb)
                else:
                    x = block(x)
            x = upsampler(x)
        
        return self.final(x)
