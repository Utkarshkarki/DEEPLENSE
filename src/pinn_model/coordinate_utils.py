"""
Coordinate utilities for the physics encoder.

Single source of truth for:
- θ grid (normalized coordinates)
- Polar encoding (r, sinφ, cosφ)
- All coordinate transformations

IMPORTANT: θ grid is created ONCE and reused everywhere
to ensure perfect spatial consistency.
"""

import torch
import torch.nn as nn

from . import config


class CoordinateSystem(nn.Module):
    """
    Creates and manages the coordinate system for the lens equation.
    
    All coordinates are normalized to [-1, 1].
    Polar encoding uses sin(φ)/cos(φ) to avoid atan2 discontinuity.
    """
    
    def __init__(self, image_size=None, device=None):
        super().__init__()
        self.image_size = image_size or config.IMAGE_SIZE
        self.device = device or config.DEVICE
        
        # Create θ grid ONCE — (2, H, W) with x,y ∈ [-1, 1]
        theta_y, theta_x = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size),
            torch.linspace(-1, 1, self.image_size),
            indexing='ij'
        )
        
        # Register as buffers (moved with model, not trained)
        self.register_buffer('theta_x', theta_x)  # (H, W)
        self.register_buffer('theta_y', theta_y)   # (H, W)
        
        # Precompute polar coordinates
        r = torch.sqrt(theta_x ** 2 + theta_y ** 2)
        r_normalized = r / r.max()  # Normalize to [0, 1]
        
        # sin(φ), cos(φ) — no atan2 discontinuity!
        phi = torch.atan2(theta_y, theta_x)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        self.register_buffer('r', r)                     # (H, W)
        self.register_buffer('r_normalized', r_normalized) # (H, W)
        self.register_buffer('sin_phi', sin_phi)           # (H, W)
        self.register_buffer('cos_phi', cos_phi)           # (H, W)
        
        # |θ| for SIS potential
        self.register_buffer('theta_abs', r)  # |θ| = r = √(x² + y²)
        
        # Grid for grid_sample: (H, W, 2) with (x, y)
        grid = torch.stack([theta_x, theta_y], dim=-1)  # (H, W, 2)
        self.register_buffer('base_grid', grid)
    
    def get_theta_grid(self):
        """Returns the base θ grid for lens equation. Shape: (H, W, 2)."""
        return self.base_grid
    
    def get_polar_encoding(self, batch_size):
        """
        Returns polar coordinate channels for input augmentation.
        
        Returns:
            (B, 3, H, W) tensor: [r_normalized, sin(φ), cos(φ)]
        """
        polar = torch.stack([
            self.r_normalized,
            self.sin_phi,
            self.cos_phi
        ], dim=0)  # (3, H, W)
        
        return polar.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    def get_theta_abs(self):
        """Returns |θ| for SIS potential. Shape: (H, W)."""
        return self.theta_abs
    
    def get_center_weight_map(self, sigma=0.5):
        """
        Gaussian weight map for center-focused consistency loss.
        w = exp(-r² / 2σ²)
        
        Args:
            sigma: Width of Gaussian (in normalized coords)
            
        Returns:
            (H, W) weight map, higher near center
        """
        w = torch.exp(-self.r ** 2 / (2 * sigma ** 2))
        return w / w.sum()  # Normalize
