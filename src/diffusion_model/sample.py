"""
Sampling, visualization, and image generation utilities.

Includes:
- Generate and save sample grids
- Denoising progression visualization
- Real vs Generated comparison
- Diversity grid
- Noise interpolation strip
- Training progress snapshots
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config
from .diffusion import GaussianDiffusion


def generate_samples(model, n_samples=16, diffusion=None):
    """Generate samples using trained DDPM model."""
    model.eval()
    diffusion = diffusion or GaussianDiffusion()
    
    with torch.no_grad():
        samples = diffusion.sample(model, n_samples)
    
    return samples


def visualize_denoising_progression(model, diffusion=None, save_path=None):
    """
    Visualize the denoising process step by step.
    
    Shows: pure noise → t=800 → t=500 → t=200 → t=50 → final image
    This demonstrates HOW the model works, not just WHAT it produces.
    """
    model.eval()
    diffusion = diffusion or GaussianDiffusion()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "denoising_progression.png")
    
    print("Generating denoising progression...")
    
    intermediate_steps = [800, 500, 200, 50]
    
    with torch.no_grad():
        _, intermediates = diffusion.sample(
            model, n_samples=4, 
            return_intermediates=True,
            intermediate_steps=intermediate_steps
        )
    
    steps = [config.TIMESTEPS] + intermediate_steps + [0]
    step_labels = [f"t={s}" if s > 0 else "Final" for s in steps]
    step_labels[0] = "Pure Noise"
    
    n_samples = 4
    n_steps = len(steps)
    
    fig, axes = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2.5, n_samples * 2.5))
    fig.suptitle("Denoising Progression: From Noise to Lensing Image", fontsize=14, y=1.02)
    
    for col, (step, label) in enumerate(zip(steps, step_labels)):
        images = intermediates[step]
        images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
        
        for row in range(n_samples):
            axes[row, col].imshow(images[row, 0], cmap='hot', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(label, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Denoising progression saved to {save_path}")


def visualize_real_vs_generated(model, train_loader, diffusion=None, save_path=None):
    """
    Side-by-side comparison: Real training images vs Generated images.
    """
    model.eval()
    diffusion = diffusion or GaussianDiffusion()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "real_vs_generated.png")
    
    print("Generating real vs generated comparison...")
    
    # Get real images
    real_batch = next(iter(train_loader))[:16]
    real_images = (real_batch.clamp(-1, 1) + 1) / 2
    
    # Generate images
    with torch.no_grad():
        gen_images = diffusion.sample(model, n_samples=16)
    gen_images = (gen_images.cpu().clamp(-1, 1) + 1) / 2
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle("Real Images (Left 4 cols) vs Generated Images (Right 4 cols)", fontsize=14)
    
    for i in range(4):
        for j in range(4):
            # Real images (left side)
            idx = i * 4 + j
            axes[i, j].imshow(real_images[idx, 0], cmap='hot', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0 and j == 0:
                axes[i, j].set_title("Real", fontsize=11)
            
            # Generated images (right side)
            axes[i, j + 4].imshow(gen_images[idx, 0], cmap='hot', vmin=0, vmax=1)
            axes[i, j + 4].axis('off')
            if i == 0 and j == 0:
                axes[i, j + 4].set_title("Generated", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Real vs generated saved to {save_path}")


def visualize_diversity_grid(model, diffusion=None, save_path=None):
    """
    Generate a 4x8 grid of 32 diverse samples.
    Demonstrates the model can produce varied outputs (not mode-collapsing).
    """
    model.eval()
    diffusion = diffusion or GaussianDiffusion()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "diversity_grid.png")
    
    print("Generating diversity grid (32 samples)...")
    
    with torch.no_grad():
        samples = diffusion.sample(model, n_samples=32)
    samples = (samples.cpu().clamp(-1, 1) + 1) / 2
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle("Diversity Grid: 32 Independent Generated Samples", fontsize=14)
    
    for i in range(4):
        for j in range(8):
            idx = i * 8 + j
            axes[i, j].imshow(samples[idx, 0], cmap='hot', vmin=0, vmax=1)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diversity grid saved to {save_path}")


def visualize_interpolation(model, diffusion=None, save_path=None, n_steps=None):
    """
    Noise space interpolation between two samples.
    Shows smooth transitions → model learned a continuous representation.
    """
    model.eval()
    diffusion = diffusion or GaussianDiffusion()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "interpolation.png")
    n_steps = n_steps or config.INTERP_STEPS
    
    print(f"Generating interpolation ({n_steps} steps)...")
    
    with torch.no_grad():
        images = diffusion.interpolate(model, n_steps=n_steps)
    images = (images.cpu().clamp(-1, 1) + 1) / 2
    
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2.5, 3))
    fig.suptitle("Noise Space Interpolation (SLERP)", fontsize=14, y=1.05)
    
    for i in range(n_steps):
        axes[i].imshow(images[i, 0], cmap='hot', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f"α={i/(n_steps-1):.1f}", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Interpolation saved to {save_path}")


def visualize_vae_vs_ddpm(ddpm_model, vae_model, diffusion=None, save_path=None):
    """
    Compare VAE and DDPM generated samples side by side.
    """
    ddpm_model.eval()
    vae_model.eval()
    diffusion = diffusion or GaussianDiffusion()
    save_path = save_path or os.path.join(config.RESULTS_DIR, "vae_vs_ddpm.png")
    
    print("Generating VAE vs DDPM comparison...")
    
    with torch.no_grad():
        ddpm_samples = diffusion.sample(ddpm_model, n_samples=8)
        vae_samples = vae_model.generate(8)
    
    ddpm_samples = (ddpm_samples.cpu().clamp(-1, 1) + 1) / 2
    vae_samples = (vae_samples.cpu().clamp(-1, 1) + 1) / 2
    
    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    fig.suptitle("VAE (Top) vs DDPM (Bottom)", fontsize=14)
    
    for j in range(8):
        axes[0, j].imshow(vae_samples[j, 0], cmap='hot', vmin=0, vmax=1)
        axes[0, j].axis('off')
        
        axes[1, j].imshow(ddpm_samples[j, 0], cmap='hot', vmin=0, vmax=1)
        axes[1, j].axis('off')
    
    axes[0, 0].set_ylabel("VAE", fontsize=12)
    axes[1, 0].set_ylabel("DDPM", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  VAE vs DDPM saved to {save_path}")
