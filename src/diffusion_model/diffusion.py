"""
Gaussian Diffusion Process for DDPM.

Implements:
- Cosine noise schedule (Nichol & Dhariwal 2021)
- Linear noise schedule (Ho et al. 2020)
- Forward diffusion (adding noise)
- Reverse diffusion (denoising / sampling)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from . import config


class GaussianDiffusion:
    """
    Gaussian diffusion process with configurable noise schedule.
    
    Args:
        timesteps: Number of diffusion steps T
        schedule_type: "cosine" or "linear"
        device: torch device
    """
    
    def __init__(
        self,
        timesteps=None,
        schedule_type=None,
        device=None
    ):
        self.timesteps = timesteps or config.TIMESTEPS
        self.schedule_type = schedule_type or config.SCHEDULE_TYPE
        self.device = device or config.DEVICE
        
        # Compute noise schedule
        if self.schedule_type == "cosine":
            betas = self._cosine_schedule()
        elif self.schedule_type == "linear":
            betas = self._linear_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Precompute diffusion quantities
        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.device),
            self.alphas_cumprod[:-1]
        ])
        
        # Quantities for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Quantities for reverse process q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_schedule(self):
        """
        Cosine noise schedule (Nichol & Dhariwal, 2021).
        
        Preserves more image structure than linear schedule,
        especially beneficial for smaller datasets.
        ᾱ_t = cos²(π/2 · (t/T + s) / (1 + s))
        """
        s = config.COSINE_S
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps)
        
        f_t = torch.cos(((t / self.timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)
        
        return betas
    
    def _linear_schedule(self):
        """Linear noise schedule (Ho et al., 2020)."""
        return torch.linspace(config.BETA_START, config.BETA_END, self.timesteps)
    
    def _extract(self, tensor, t, shape):
        """Extract values from tensor at timestep t, broadcast to shape."""
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
    
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward process: add noise to image.
        
        q(x_t | x_0) = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        
        Args:
            x_0: Clean image (B, C, H, W), range [-1, 1]
            t: Timestep (B,)
            noise: Optional pre-sampled noise
            
        Returns:
            x_t: Noisy image
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        return x_t, noise
    
    def compute_loss(self, model, x_0):
        """
        Compute DDPM training loss.
        
        L = E[||ε - ε_θ(x_t, t)||²]
        
        Args:
            model: U-Net noise prediction model
            x_0: Clean images (B, C, H, W)
            
        Returns:
            loss: MSE loss between true and predicted noise
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise and create noisy image
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_diffusion(x_0, t, noise)
        
        # Predict noise
        predicted_noise = model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, model, n_samples, channels=None, image_size=None, 
               return_intermediates=False, intermediate_steps=None):
        """
        Reverse process: generate images from noise.
        
        Args:
            model: Trained U-Net
            n_samples: Number of images to generate
            channels: Number of channels
            image_size: Image size
            return_intermediates: If True, return intermediate denoising steps
            intermediate_steps: List of timesteps to save (e.g., [800, 500, 200, 50])
            
        Returns:
            samples: Generated images (B, C, H, W), range [-1, 1]
            intermediates: (optional) dict of timestep → images
        """
        channels = channels or config.CHANNELS
        image_size = image_size or config.IMAGE_SIZE
        
        if intermediate_steps is None:
            intermediate_steps = [800, 500, 200, 50]
        
        intermediates = {}
        
        # Start from pure noise
        x = torch.randn(n_samples, channels, image_size, image_size, device=self.device)
        
        if return_intermediates:
            intermediates[self.timesteps] = x.clone().cpu()
        
        # Reverse diffusion
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((n_samples,), t_val, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t)
            
            # Compute x_{t-1}
            alpha = self._extract(self.alphas, t, x.shape)
            alpha_cumprod = self._extract(self.alphas_cumprod, t, x.shape)
            sqrt_one_minus_alpha_cumprod = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            
            # Mean of q(x_{t-1} | x_t, x_0)
            x = (1.0 / torch.sqrt(alpha)) * (
                x - (1.0 - alpha) / sqrt_one_minus_alpha_cumprod * predicted_noise
            )
            
            # Add noise for t > 0
            if t_val > 0:
                noise = torch.randn_like(x)
                posterior_var = self._extract(self.posterior_variance, t, x.shape)
                x = x + torch.sqrt(posterior_var) * noise
            
            # Save intermediates
            if return_intermediates and t_val in intermediate_steps:
                intermediates[t_val] = x.clone().cpu()
        
        if return_intermediates:
            intermediates[0] = x.clone().cpu()
            return x.clamp(-1, 1), intermediates
        
        return x.clamp(-1, 1)
    
    @torch.no_grad()
    def interpolate(self, model, n_steps=None):
        """
        Noise space interpolation between two samples.
        
        Generates images along a path between two random noise vectors,
        demonstrating the model learned a continuous latent space.
        
        Args:
            model: Trained U-Net
            n_steps: Number of interpolation steps
            
        Returns:
            images: (n_steps, C, H, W) interpolated generated images
        """
        n_steps = n_steps or config.INTERP_STEPS
        
        # Two random noise endpoints
        z1 = torch.randn(1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE, 
                         device=self.device)
        z2 = torch.randn(1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE, 
                         device=self.device)
        
        images = []
        for alpha in torch.linspace(0, 1, n_steps):
            # Spherical interpolation (slerp) for better results
            z = slerp(alpha.item(), z1, z2)
            
            # Denoise from interpolated noise
            x = z.clone()
            for t_val in reversed(range(self.timesteps)):
                t = torch.full((1,), t_val, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t)
                
                a = self._extract(self.alphas, t, x.shape)
                ac = self._extract(self.alphas_cumprod, t, x.shape)
                s = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                
                x = (1.0 / torch.sqrt(a)) * (x - (1.0 - a) / s * predicted_noise)
                
                if t_val > 0:
                    pv = self._extract(self.posterior_variance, t, x.shape)
                    x = x + torch.sqrt(pv) * torch.randn_like(x)
            
            images.append(x.clamp(-1, 1).cpu())
        
        return torch.cat(images, dim=0)


def slerp(t, v0, v1):
    """
    Spherical linear interpolation between two tensors.
    Better than linear interpolation for high-dimensional noise vectors.
    """
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    # Compute angle between vectors
    dot = torch.sum(v0_flat * v1_flat) / (torch.norm(v0_flat) * torch.norm(v1_flat))
    dot = torch.clamp(dot, -1, 1)
    omega = torch.acos(dot)
    
    if omega.abs() < 1e-6:
        # Vectors are nearly parallel, use linear interpolation
        return (1.0 - t) * v0 + t * v1
    
    sin_omega = torch.sin(omega)
    result = (torch.sin((1.0 - t) * omega) / sin_omega) * v0 + \
             (torch.sin(t * omega) / sin_omega) * v1
    
    return result
