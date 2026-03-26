"""
Training loop for Consistency-Regularized Lensiformer.

Features:
- Multi-loss: L_class + L_physics + L_k + L_res + L_alpha
- Curriculum learning for physics loss
- Dynamic loss balancing (normalize by running mean)
- LR warmup + cosine annealing
- Checkpoint saving, validation tracking
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config
from .dataset import get_dataloaders
from .lensiformer import Lensiformer
from .baseline import ResNet18Baseline, PhysicsCNN


class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss with curriculum learning.
    
    L = L_class + λ₁·L_physics + λ₂·L_k + λ₃·L_res + λ₄·L_alpha
    
    All λ follow a curriculum: 0 → target over epochs.
    All losses normalized by running mean for stable gradients.
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Pre-create CoordinateSystem ONCE (not per-batch!)
        from .coordinate_utils import CoordinateSystem
        self.coords = CoordinateSystem()
        
        # Running means for dynamic balancing
        self.running_means = {
            'class': 1.0, 'physics': 1.0,
            'k': 1.0, 'res': 1.0, 'alpha': 1.0
        }
        self.momentum = 0.99
    
    def get_physics_weight(self, epoch):
        """Curriculum schedule for ALL auxiliary loss weights."""
        if epoch < config.CURRICULUM_START:
            return 0.0
        elif epoch < config.CURRICULUM_END:
            # Linear ramp
            progress = (epoch - config.CURRICULUM_START) / (config.CURRICULUM_END - config.CURRICULUM_START)
            return progress
        else:
            return 1.0
    
    def _update_running_mean(self, key, value):
        self.running_means[key] = (
            self.momentum * self.running_means[key] + (1 - self.momentum) * value
        )
    
    def forward(self, logits, labels, physics_data, image, epoch):
        """
        Compute total loss.
        
        Args:
            logits: (B, C) model predictions
            labels: (B,) ground truth
            physics_data: dict from inverse lens layer
            image: (B, 1, H, W) original input
            epoch: current epoch for curriculum
        """
        losses = {}
        
        # 1. Classification loss
        l_class = self.ce_loss(logits, labels)
        self._update_running_mean('class', l_class.item())
        losses['class'] = l_class
        
        # Skip physics losses if no physics data (baseline models)
        if not physics_data:
            losses['total'] = l_class
            return losses
        
        lambda_phys = self.get_physics_weight(epoch)
        
        # 2. Physics consistency loss: ||Î − I||² (center-weighted)
        if lambda_phys > 0:
            import torch.nn.functional as F
            source = physics_data['source']
            
            # Forward lens: re-lens source using SAME α
            forward_grid = physics_data['base_grid'] + physics_data['alpha_grid']
            forward_grid = torch.clamp(forward_grid, -1.0, 1.0)
            
            reconstructed = F.grid_sample(
                source, forward_grid, mode='bilinear',
                padding_mode='border', align_corners=True
            )
            
            # Center-weighted MSE — reuse pre-created coords (no per-batch allocation!)
            self.coords = self.coords.to(image.device)
            weight_map = self.coords.get_center_weight_map(sigma=0.5)
            weight_map = weight_map.unsqueeze(0).unsqueeze(0)
            
            l_physics = (weight_map * (reconstructed - image) ** 2).mean()
            self._update_running_mean('physics', l_physics.item())
            losses['physics'] = l_physics
        else:
            l_physics = torch.tensor(0.0, device=image.device)
            losses['physics'] = l_physics
        
        # 3. k-smoothness loss: ||∇k||²
        k = physics_data['k']
        dk_dx = k[:, :, :, 1:] - k[:, :, :, :-1]
        dk_dy = k[:, :, 1:, :] - k[:, :, :-1, :]
        l_k = (dk_dx ** 2).mean() + (dk_dy ** 2).mean()
        self._update_running_mean('k', l_k.item())
        losses['k_smooth'] = l_k
        
        # 4. ψ_residual penalty: ||ψ_res||²
        l_res = (physics_data['psi_res'] ** 2).mean()
        self._update_running_mean('res', l_res.item())
        losses['res'] = l_res
        
        # 5. α energy constraint: ||α||²
        l_alpha = (physics_data['alpha_x'] ** 2 + physics_data['alpha_y'] ** 2).mean()
        self._update_running_mean('alpha', l_alpha.item())
        losses['alpha'] = l_alpha
        
        # Dynamic balancing: normalize each loss by running mean
        # ALL auxiliary losses obey the same curriculum — no aux-loss interference
        # during the critical early classification-learning phase.
        total = l_class
        if lambda_phys > 0:
            total = total + (config.LAMBDA_PHYSICS * lambda_phys) * (l_physics / max(self.running_means['physics'], 1e-8))
            total = total + (config.LAMBDA_K_SMOOTH * lambda_phys) * (l_k / max(self.running_means['k'], 1e-8))
            total = total + (config.LAMBDA_RES * lambda_phys) * (l_res / max(self.running_means['res'], 1e-8))
            total = total + (config.LAMBDA_ALPHA * lambda_phys) * (l_alpha / max(self.running_means['alpha'], 1e-8))
        
        losses['total'] = total
        
        return losses


def train_model(
    model_type="lensiformer",
    num_epochs=None,
    data_fraction=1.0,
    experiment_name=None
):
    """
    Train a model.
    
    Args:
        model_type: "lensiformer", "resnet", or "physics_cnn"
        num_epochs: Number of epochs
        data_fraction: Fraction of training data (for data efficiency test)
        experiment_name: Name for saving results
        
    Returns:
        model, history dict
    """
    num_epochs = num_epochs or config.NUM_EPOCHS
    experiment_name = experiment_name or model_type
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"  Model: {model_type}, Epochs: {num_epochs}")
    print(f"  Data fraction: {data_fraction*100:.0f}%")
    print(f"  Device: {config.DEVICE}")
    print(f"{'='*60}\n")
    
    # Data
    train_loader, val_loader = get_dataloaders(data_fraction=data_fraction)
    
    # Model
    if model_type == "lensiformer":
        model = Lensiformer().to(config.DEVICE)
    elif model_type == "resnet":
        model = ResNet18Baseline().to(config.DEVICE)
    elif model_type == "physics_cnn":
        model = PhysicsCNN().to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    has_physics = model_type in ["lensiformer", "physics_cnn"]
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")
    
    # Loss, optimizer, scheduler
    criterion = PhysicsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    
    # LR warmup + cosine annealing
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=config.WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(num_epochs - config.WARMUP_EPOCHS, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[config.WARMUP_EPOCHS])
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'loss_class': [], 'loss_physics': [], 'loss_k': [],
        'loss_res': [], 'loss_alpha': [], 'lr': []
    }
    
    best_val_acc = 0.0
    start_epoch = 1

    # Auto-resume from latest checkpoint (exact epoch) or best
    latest_path = os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_latest.pth")
    best_path = os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_best.pth")
    ckpt_path = latest_path if os.path.exists(latest_path) else best_path
    if os.path.exists(ckpt_path):
        print(f"\nResuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', ckpt.get('val_acc', 0.0))
        if 'history' in ckpt:
            history = ckpt['history']
        # Restore optimizer & scheduler if saved (backward-compatible)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Only restore scheduler if total epochs match — T_max is baked into scheduler
        # state and would corrupt LR if mismatched (e.g. 15-epoch checkpoint → 100-epoch run)
        ckpt_num_epochs = ckpt.get('num_epochs', None)
        if 'scheduler_state_dict' in ckpt and ckpt_num_epochs == num_epochs:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        elif 'scheduler_state_dict' in ckpt:
            print(f"  [Note] Skipping scheduler state restore: checkpoint was trained for "
                  f"{ckpt_num_epochs} epochs, current run is {num_epochs} epochs. "
                  f"LR schedule will restart from epoch {start_epoch}.")
        print(f"Resuming from Epoch {start_epoch}, Best Val Acc so far: {best_val_acc:.1f}%\n")

    for epoch in range(start_epoch, num_epochs + 1):
        # ─── Train ──────────────────────────────────────────────
        model.train()
        epoch_losses = {k: 0.0 for k in ['total', 'class', 'physics', 'k_smooth', 'res', 'alpha']}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Forward
            if has_physics:
                logits, physics_data = model(images, return_physics=True)
            else:
                logits = model(images)
                physics_data = {}
            
            # Loss
            losses = criterion(logits, labels, physics_data, images, epoch)
            
            # Skip batch if loss is NaN/Inf (numerical instability in physics ops)
            if not torch.isfinite(losses['total']):
                optimizer.zero_grad()
                continue
            
            # Backward — catch CUDA errors from bad batches instead of crashing
            try:
                optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except RuntimeError as e:
                if 'CUDA' in str(e):
                    print(f"\n  [Warning] CUDA error on batch, skipping: {e}")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                raise
            
            # Track
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            
            pbar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                cls=f"{losses['class'].item():.4f}"
            )
        
        scheduler.step()
        
        # Average losses
        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        # ─── Validate ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                
                logits = model(images) if not has_physics else model(images, return_physics=False)
                
                loss = nn.CrossEntropyLoss()(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total
        
        # Log
        history['train_loss'].append(epoch_losses['total'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['loss_class'].append(epoch_losses['class'])
        history['loss_physics'].append(epoch_losses.get('physics', 0))
        history['loss_k'].append(epoch_losses.get('k_smooth', 0))
        history['loss_res'].append(epoch_losses.get('res', 0))
        history['loss_alpha'].append(epoch_losses.get('alpha', 0))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        lambda_phys = criterion.get_physics_weight(epoch)
        print(f"Epoch {epoch}: train={epoch_losses['total']:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% "
              f"lambda_phys={lambda_phys:.3f} lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'num_epochs': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'history': history,
            }, os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_best.pth"))

        # Always save latest checkpoint so resume starts from exact last epoch
        torch.save({
            'epoch': epoch,
            'num_epochs': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'history': history,
        }, os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_latest.pth"))

        # Periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(config.CHECKPOINT_DIR, f"{experiment_name}_epoch{epoch}.pth"))
    
    # Plot training curves
    _plot_training_curves(history, experiment_name, has_physics)
    
    print(f"\nTraining complete! Best val acc: {best_val_acc:.1f}%")
    return model, history


def _plot_training_curves(history, name, has_physics=True):
    """Plot and save training curves."""
    save_dir = config.RESULTS_DIR
    
    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=1.5)
    axes[0].plot(history['val_loss'], label='Val', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{name} — Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['val_acc'], 'g-', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{name} — Validation Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Physics loss breakdown (if applicable)
    if has_physics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(history['loss_class'], label='L_class', linewidth=1.5)
        ax.plot(history['loss_physics'], label='L_physics', linewidth=1.5)
        ax.plot(history['loss_k'], label='L_k', linewidth=1.5)
        ax.plot(history['loss_res'], label='L_res', linewidth=1.5)
        ax.plot(history['loss_alpha'], label='L_alpha', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} — Physics Loss Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_physics_losses.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Curves saved to {save_dir}")
