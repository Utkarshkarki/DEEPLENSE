"""
Ablation study runner.

Runs 2 ablation experiments to study what components matter:
1. With Attention vs Without Attention
2. With EMA vs Without EMA

Each ablation trains for fewer epochs and reports FID + sample grids.
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config
from .train import train_ddpm
from .diffusion import GaussianDiffusion
from .evaluate import calculate_fid
from .dataset import get_dataloaders


def run_ablations(full_model=None, full_losses=None):
    """
    Run all ablation experiments and compile results.
    
    Args:
        full_model: Already-trained full DDPM model (optional, avoids retraining)
        full_losses: Losses from full model training
        
    Returns:
        results: dict with ablation results
    """
    ablation_dir = os.path.join(config.RESULTS_DIR, "ablations")
    os.makedirs(ablation_dir, exist_ok=True)
    
    diffusion = GaussianDiffusion()
    results = {}
    all_losses = {}
    
    # ─── Ablation 1: No Attention ───────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION 1: Without Self-Attention")
    print("=" * 60)
    
    model_no_attn, losses_no_attn = train_ddpm(
        num_epochs=config.ABLATION_EPOCHS,
        use_attention=False,
        use_ema=True,
        experiment_name="ablation_no_attention",
        save_dir=ablation_dir
    )
    all_losses['No Attention'] = losses_no_attn
    
    # ─── Ablation 2: No EMA ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION 2: Without EMA")
    print("=" * 60)
    
    model_no_ema, losses_no_ema = train_ddpm(
        num_epochs=config.ABLATION_EPOCHS,
        use_attention=True,
        use_ema=False,
        experiment_name="ablation_no_ema",
        save_dir=ablation_dir
    )
    all_losses['No EMA'] = losses_no_ema
    
    # ─── Compute FID for ablations ──────────────────────────────────
    print("\nComputing FID scores for ablation models...")
    
    train_loader, _ = get_dataloaders()
    
    # Collect real images
    real_list = []
    for batch in train_loader:
        real_list.append(batch)
        if sum(b.shape[0] for b in real_list) >= config.FID_NUM_SAMPLES:
            break
    real_images = torch.cat(real_list, dim=0)[:config.FID_NUM_SAMPLES]
    real_images_01 = (real_images.clamp(-1, 1) + 1) / 2
    
    # Generate samples and compute FID for each ablation
    for name, model in [("No Attention", model_no_attn), ("No EMA", model_no_ema)]:
        model.eval()
        gen_list = []
        for i in range(0, config.FID_NUM_SAMPLES, 32):
            n = min(32, config.FID_NUM_SAMPLES - i)
            with torch.no_grad():
                samples = diffusion.sample(model, n_samples=n)
            gen_list.append(samples.cpu())
        gen_images = torch.cat(gen_list, dim=0)
        gen_images_01 = (gen_images.clamp(-1, 1) + 1) / 2
        
        fid = calculate_fid(real_images_01, gen_images_01)
        results[name] = fid
    
    # ─── Compare loss curves ────────────────────────────────────────
    if full_losses is not None:
        all_losses['Full DDPM'] = full_losses[:config.ABLATION_EPOCHS]
    
    _plot_ablation_losses(all_losses, os.path.join(ablation_dir, "ablation_losses.png"))
    _plot_ablation_fid(results, os.path.join(ablation_dir, "ablation_fid.png"))
    
    # Print summary
    print(f"\n{'='*50}")
    print("ABLATION RESULTS")
    print(f"{'='*50}")
    for name, fid in results.items():
        print(f"  {name:20s}: FID = {fid:.2f}")
    print(f"{'='*50}\n")
    
    return results


def _plot_ablation_losses(losses_dict, save_path):
    """Plot loss curves for all ablation experiments."""
    plt.figure(figsize=(10, 6))
    
    colors = {'Full DDPM': 'blue', 'No Attention': 'red', 'No EMA': 'green'}
    
    for name, losses in losses_dict.items():
        color = colors.get(name, 'gray')
        plt.plot(range(1, len(losses) + 1), losses, label=name, 
                linewidth=1.5, color=color)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Ablation Study: Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation losses saved to {save_path}")


def _plot_ablation_fid(results, save_path):
    """Bar chart comparing FID scores across ablation variants."""
    names = list(results.keys())
    fids = list(results.values())
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, fids, color=['#e74c3c', '#2ecc71'], width=0.5, edgecolor='black')
    
    # Add value labels on bars
    for bar, fid in zip(bars, fids):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{fid:.1f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.ylabel('FID Score (lower = better)', fontsize=12)
    plt.title('Ablation Study: FID Score Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation FID chart saved to {save_path}")
