"""
Training loops for DDPM and VAE.

Features:
- DDPM training with EMA
- VAE baseline training
- Loss logging, checkpoint saving, sample generation during training
"""

import os
import copy
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config
from .dataset import get_dataloaders
from .unet import UNet
from .vae import ConvVAE, vae_loss
from .diffusion import GaussianDiffusion


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights (for sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights (for training)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_ddpm(
    num_epochs=None,
    use_attention=True,
    use_ema=True,
    experiment_name="ddpm",
    save_dir=None
):
    """
    Train DDPM model.
    
    Args:
        num_epochs: Number of training epochs
        use_attention: Whether U-Net uses self-attention (for ablation)
        use_ema: Whether to use EMA (for ablation)
        experiment_name: Name for saving checkpoints/results
        save_dir: Override save directory
        
    Returns:
        model: Trained model
        losses: List of epoch losses
    """
    num_epochs = num_epochs or config.NUM_EPOCHS
    save_dir = save_dir or config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training DDPM: {experiment_name}")
    print(f"  Epochs: {num_epochs}, Attention: {use_attention}, EMA: {use_ema}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Schedule: {config.SCHEDULE_TYPE}")
    print(f"{'='*60}\n")
    
    # Data
    train_loader, val_loader = get_dataloaders()
    
    # Model
    model = UNet(use_attention=use_attention).to(config.DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"U-Net parameters: {param_count:,}")
    
    # Diffusion
    diffusion = GaussianDiffusion()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # EMA
    ema = EMA(model, config.EMA_DECAY) if use_ema else None
    
    # Training loop
    losses = []
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(config.DEVICE)
            
            # Compute loss
            loss = diffusion.compute_loss(model, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # EMA update
            if ema is not None:
                ema.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch}: avg loss = {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'ema_shadow': ema.shadow if ema else None,
            }, os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_best.pth"))
        
        # Periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'ema_shadow': ema.shadow if ema else None,
            }, os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_epoch{epoch}.pth"))
        
        # Generate samples periodically
        if epoch % config.SAMPLE_INTERVAL == 0:
            model.eval()
            if ema is not None:
                ema.apply_shadow()
            
            samples = diffusion.sample(model, n_samples=16)
            _save_sample_grid(
                samples, 
                os.path.join(save_dir, f"{experiment_name}_epoch{epoch}.png"),
                title=f"{experiment_name} - Epoch {epoch}"
            )
            
            if ema is not None:
                ema.restore()
    
    # Save loss curve
    _plot_loss_curve(losses, os.path.join(save_dir, f"{experiment_name}_loss.png"), experiment_name)
    
    # Apply EMA for final model
    if ema is not None:
        ema.apply_shadow()
    
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    return model, losses


def train_vae(num_epochs=None, experiment_name="vae"):
    """
    Train VAE baseline model.
    
    Args:
        num_epochs: Number of training epochs
        experiment_name: Name for saving
        
    Returns:
        model: Trained VAE
        losses: List of epoch losses
    """
    num_epochs = num_epochs or config.VAE_EPOCHS
    
    print(f"\n{'='*60}")
    print(f"Training VAE Baseline")
    print(f"  Epochs: {num_epochs}, Latent dim: {config.VAE_LATENT_DIM}")
    print(f"  Device: {config.DEVICE}")
    print(f"{'='*60}\n")
    
    train_loader, _ = get_dataloaders()
    
    model = ConvVAE().to(config.DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {param_count:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.VAE_LR)
    
    losses = []
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(config.DEVICE)
            
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"VAE Epoch {epoch}: loss = {avg_loss:.6f}")
        
        # Generate samples periodically
        if epoch % config.SAMPLE_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                samples = model.generate(16) 
            _save_sample_grid(
                samples,
                os.path.join(config.RESULTS_DIR, f"{experiment_name}_epoch{epoch}.png"),
                title=f"VAE - Epoch {epoch}"
            )
    
    # Save model
    torch.save(model.state_dict(), 
               os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_final.pth"))
    
    # Save loss curve
    _plot_loss_curve(losses, os.path.join(config.RESULTS_DIR, f"{experiment_name}_loss.png"), "VAE")
    
    print(f"\nVAE training complete!")
    return model, losses


def _save_sample_grid(samples, path, title="Generated Samples", nrow=4):
    """Save a grid of generated images."""
    # Convert from [-1,1] to [0,1]
    samples = (samples.cpu().clamp(-1, 1) + 1) / 2
    n = min(samples.shape[0], nrow * nrow)
    
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    
    for i in range(nrow):
        for j in range(nrow):
            idx = i * nrow + j
            if idx < n:
                axes[i, j].imshow(samples[idx, 0], cmap='hot', vmin=0, vmax=1)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Samples saved to {path}")


def _plot_loss_curve(losses, path, name="DDPM"):
    """Plot and save training loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{name} Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Loss curve saved to {path}")


if __name__ == "__main__":
    train_ddpm()
