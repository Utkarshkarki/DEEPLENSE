"""
Convolutional VAE baseline for comparison with DDPM.

A simple VAE provides a baseline to demonstrate that DDPM produces
sharper, more diverse images — showing research thinking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for gravitational lensing images.
    
    Encoder: Conv layers → flatten → μ, log σ²
    Decoder: Linear → reshape → ConvTranspose layers
    """
    
    def __init__(self, in_channels=None, latent_dim=None, image_size=None):
        super().__init__()
        
        self.in_channels = in_channels or config.CHANNELS
        self.latent_dim = latent_dim or config.VAE_LATENT_DIM
        self.image_size = image_size or config.IMAGE_SIZE
        
        # ─── Encoder ────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # (1, 128, 128) → (32, 64, 64)
            nn.Conv2d(self.in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # (32, 64, 64) → (64, 32, 32)
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 32, 32) → (128, 16, 16)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 16, 16) → (256, 8, 8)
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        
        self.flatten_size = 256 * 8 * 8  # After encoding 128x128
        
        self.fc_mu = nn.Linear(self.flatten_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, self.latent_dim)
        
        # ─── Decoder ────────────────────────────────────────────────
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # (256, 8, 8) → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # (128, 16, 16) → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 32, 32) → (32, 64, 64)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # (32, 64, 64) → (1, 128, 128)
            nn.ConvTranspose2d(32, self.in_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    @torch.no_grad()
    def generate(self, n_samples, device=None):
        """Generate new images by sampling from the prior."""
        device = device or config.DEVICE
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon_x, x, mu, logvar, kl_weight=None):
    """
    VAE loss = Reconstruction + KL divergence.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Latent mean
        logvar: Latent log-variance
        kl_weight: Weight for KL term (β-VAE style)
    """
    kl_weight = kl_weight or config.VAE_KL_WEIGHT
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence: D_KL(q(z|x) || p(z))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss
