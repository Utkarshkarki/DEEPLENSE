"""
Physics operations for the Lensiformer.

Implements differentiable versions of:
- Gaussian blur for potential smoothing
- Spatial gradients for deflection computation
- SIS (Singular Isothermal Sphere) potential
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import config


class GaussianBlur2D(nn.Module):
    """
    Differentiable Gaussian blur for smoothing the gravitational potential.
    
    Smoothing ψ before computing gradients prevents noisy deflection fields.
    σ is tunable — can be swept in [0.5, 2.0] for best results.
    """
    
    def __init__(self, sigma=None, kernel_size=None):
        super().__init__()
        self.sigma = sigma or config.GAUSSIAN_SIGMA
        
        # Kernel size: 6σ + 1 (covers 99.7% of distribution)
        if kernel_size is None:
            kernel_size = int(6 * self.sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # Register as buffer (not a parameter, moves with model)
        self.register_buffer('kernel', gauss_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        """
        Apply Gaussian blur.
        
        Args:
            x: (B, 1, H, W) tensor
            
        Returns:
            Blurred tensor (B, 1, H, W)
        """
        return F.conv2d(x, self.kernel, padding=self.padding)


class SpatialGradient(nn.Module):
    """
    Compute spatial gradients using Sobel-like filters.
    
    Used to compute deflection: α = ∇ψ
    
    Gradients are normalized to grid space (÷ W/2, H/2) so that
    θ, α, and β all live in the same [-1, 1] coordinate system.
    """
    
    def __init__(self, image_size=None):
        super().__init__()
        self.image_size = image_size or config.IMAGE_SIZE
        
        # Sobel kernels for dx and dy
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, psi):
        """
        Compute gradient of potential ψ.
        
        Args:
            psi: (B, 1, H, W) potential field
            
        Returns:
            alpha_x, alpha_y: (B, 1, H, W) each, normalized to grid space
        """
        # Raw gradients (in pixel space)
        grad_x = F.conv2d(psi, self.sobel_x, padding=1)
        grad_y = F.conv2d(psi, self.sobel_y, padding=1)
        
        # Normalize to [-1, 1] coordinate space
        # gradient in pixel space * (2/W) = gradient in normalized space
        scale = 2.0 / self.image_size
        grad_x = grad_x * scale
        grad_y = grad_y * scale
        
        return grad_x, grad_y


def soft_clamp(x):
    """
    Soft clamping: x / (1 + |x|)
    
    Unlike tanh, this doesn't cause vanishing gradients.
    Maps R → (-1, 1) smoothly.
    """
    return x / (1.0 + torch.abs(x))


def compute_sis_potential(theta_abs, k):
    """
    Compute SIS (Singular Isothermal Sphere) gravitational potential.
    
    ψ_SIS = k · |θ|
    
    Args:
        theta_abs: |θ| values, shape (H, W)
        k: Per-pixel correction factor, shape (B, 1, H, W)
        
    Returns:
        psi: (B, 1, H, W) potential field
    """
    return k * theta_abs.unsqueeze(0).unsqueeze(0)
