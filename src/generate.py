"""
Generate gravitational lensing images from a trained DDPM model.

Usage (Git Bash):
    venv/Scripts/python.exe src/generate.py

Images are saved to: src/results/diffusion/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from diffusion_model import config
from diffusion_model.unet import UNet
from diffusion_model.diffusion import GaussianDiffusion
from diffusion_model.sample import (
    visualize_diversity_grid,
    visualize_denoising_progression,
    visualize_interpolation,
)


def load_model(checkpoint_name="ddpm_best.pth"):
    """Load a trained DDPM model from checkpoint."""
    model = UNet().to(config.DEVICE)
    
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(ckpt_path):
        # Try quick_test checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "quick_test_best.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found in {config.CHECKPOINT_DIR}")
        print("Train the model first!")
        sys.exit(1)
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.6f}")
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("DDPM Lensing Image Generator")
    print(f"Device: {config.DEVICE}")
    print("=" * 50)
    
    model = load_model()
    diffusion = GaussianDiffusion()
    
    print("\n1/3 Generating diversity grid (32 samples)...")
    visualize_diversity_grid(model, diffusion)
    
    print("\n2/3 Generating denoising progression...")
    visualize_denoising_progression(model, diffusion)
    
    print("\n3/3 Generating noise interpolation...")
    visualize_interpolation(model, diffusion)
    
    print(f"\n{'=' * 50}")
    print(f"All images saved to: {config.RESULTS_DIR}")
    print("=" * 50)
