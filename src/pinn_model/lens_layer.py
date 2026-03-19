"""
Inverse Lens Layer — the physics heart of the Lensiformer.

Implements:
1. k-Predictor: CNN that predicts per-pixel SIS correction factors
2. ψ_residual predictor: CNN for beyond-SIS corrections
3. Inverse lens: β = θ − α (source reconstruction)
4. Forward lens: Î = warp(S, θ+α) for consistency loss (shared α!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .coordinate_utils import CoordinateSystem
from .physics_ops import GaussianBlur2D, SpatialGradient, soft_clamp, compute_sis_potential


class KPredictor(nn.Module):
    """
    Predicts per-pixel k corrections to the SIS potential.
    
    k(i,j) = k_SIS * (1 + K_RANGE * tanh(CNN(image)))
    
    Initialized near k_SIS for stable starting point.
    """
    
    def __init__(self, in_channels=4, k_sis=None, k_range=None):
        super().__init__()
        self.k_sis = k_sis or config.K_SIS_INIT
        self.k_range = k_range or config.K_RANGE
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),  # Per-pixel output
        )
        
        # Initialize near zero so k ≈ k_SIS at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) — image + polar encoding
        Returns:
            k: (B, 1, H, W) per-pixel k values near k_SIS
        """
        raw = self.net(x)
        k = self.k_sis * (1.0 + self.k_range * torch.tanh(raw))
        return k


class PsiResidualPredictor(nn.Module):
    """
    Predicts residual potential beyond SIS assumption.
    
    ψ_residual = PSI_RES_SCALE * tanh(CNN(image))
    
    Magnitude-clamped to prevent it from dominating.
    """
    
    def __init__(self, in_channels=4, scale=None):
        super().__init__()
        self.scale = scale or config.PSI_RES_SCALE
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
        
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W)
        Returns:
            psi_res: (B, 1, H, W) residual potential, magnitude-clamped
        """
        raw = self.net(x)
        return self.scale * torch.tanh(raw)


class InverseLensLayer(nn.Module):
    """
    Differentiable inverse gravitational lens layer.
    
    Given a lensed image I, reconstructs the source galaxy S using:
        ψ = k·|θ| + ψ_residual
        ψ_smooth = gaussian_blur(ψ)
        α = ∇(ψ_smooth), soft-clamped
        β = θ − α
        S = (1−w)·grid_sample(I, β) + w·I   (identity skip)
    
    Also provides forward lens for consistency loss using the SAME α.
    """
    
    def __init__(self):
        super().__init__()
        
        self.coords = CoordinateSystem()
        self.k_predictor = KPredictor(in_channels=4)
        self.psi_residual = PsiResidualPredictor(in_channels=4)
        self.gaussian_blur = GaussianBlur2D()
        self.spatial_grad = SpatialGradient()
        
        self.skip_weight = config.IDENTITY_SKIP_WEIGHT
    
    def forward(self, image):
        """
        Inverse lens: reconstruct source from lensed image.
        
        Args:
            image: (B, 1, H, W) lensed image in [0, 1]
            
        Returns:
            source: (B, 1, H, W) reconstructed source
            physics_data: dict with k, psi, alpha, source for visualization & losses
        """
        B = image.shape[0]
        
        # Build augmented input: [image, r, sinφ, cosφ]
        polar = self.coords.get_polar_encoding(B)  # (B, 3, H, W)
        x_aug = torch.cat([image, polar], dim=1)     # (B, 4, H, W)
        
        # Predict k and ψ_residual
        k = self.k_predictor(x_aug)          # (B, 1, H, W)
        psi_res = self.psi_residual(x_aug)   # (B, 1, H, W)
        
        # Compute total potential: ψ = k·|θ| + ψ_residual
        theta_abs = self.coords.get_theta_abs()  # (H, W)
        psi_sis = compute_sis_potential(theta_abs, k)  # (B, 1, H, W)
        psi = psi_sis + psi_res
        
        # Smooth potential before gradient
        psi_smooth = self.gaussian_blur(psi)
        
        # Compute deflection α = ∇ψ (already normalized to grid space)
        alpha_x, alpha_y = self.spatial_grad(psi_smooth)  # (B, 1, H, W)
        
        # Soft clamp: α / (1 + |α|) — prevents grid collapse, no vanishing gradients
        alpha_x = soft_clamp(alpha_x)
        alpha_y = soft_clamp(alpha_y)
        
        # Compute source position: β = θ − α
        base_grid = self.coords.get_theta_grid()  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        
        # Build deflection grid
        alpha_grid = torch.stack([alpha_x.squeeze(1), alpha_y.squeeze(1)], dim=-1)  # (B, H, W, 2)
        beta_grid = base_grid - alpha_grid
        
        # Clamp to valid range
        beta_grid = torch.clamp(beta_grid, -1.0, 1.0)
        
        # Warp image to reconstruct source
        warped = F.grid_sample(image, beta_grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        
        # Identity skip for gradient stability
        source = (1.0 - self.skip_weight) * warped + self.skip_weight * image
        
        # Store for consistency loss and visualization
        physics_data = {
            'k': k,
            'psi': psi,
            'psi_res': psi_res,
            'alpha_x': alpha_x,
            'alpha_y': alpha_y,
            'alpha_grid': alpha_grid,
            'source': source,
            'base_grid': base_grid,
        }
        
        return source, physics_data
    
    def forward_lens(self, source, physics_data):
        """
        Forward lens: re-lens the reconstructed source using SAME α.
        
        Î = grid_sample(S, θ + α)
        
        Uses the same α from the inverse pass for consistency.
        
        Args:
            source: (B, 1, H, W) reconstructed source
            physics_data: dict from inverse_lens containing alpha_grid and base_grid
            
        Returns:
            reconstructed_image: (B, 1, H, W)
        """
        # θ + α (reuse same α!)
        forward_grid = physics_data['base_grid'] + physics_data['alpha_grid']
        forward_grid = torch.clamp(forward_grid, -1.0, 1.0)
        
        reconstructed = F.grid_sample(source, forward_grid, mode='bilinear',
                                       padding_mode='border', align_corners=True)
        
        return reconstructed
