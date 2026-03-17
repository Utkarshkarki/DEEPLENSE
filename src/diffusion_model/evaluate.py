"""
Evaluation metrics for generated images.

Includes:
- FID (Fréchet Inception Distance) computation
- Power spectrum analysis (physics-informed)
- Pixel intensity distribution comparison
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import config


# ─── FID Score ──────────────────────────────────────────────────────────────

def calculate_fid(real_images, generated_images, device=None, batch_size=None):
    """
    Compute Fréchet Inception Distance between real and generated images.
    
    Uses InceptionV3 features. Since our images are 1-channel grayscale,
    we tile to 3 channels and resize to 299x299.
    
    Args:
        real_images: Tensor (N, 1, H, W) in [0, 1]
        generated_images: Tensor (N, 1, H, W) in [0, 1]
        
    Returns:
        fid_score: float
    """
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    
    device = device or config.DEVICE
    batch_size = batch_size or config.FID_BATCH_SIZE
    
    print("Computing FID score...")
    
    # Load InceptionV3
    inception = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove classifier, get features
    inception = inception.to(device).eval()
    
    def get_features(images):
        """Extract Inception features from images."""
        features_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            
            # Tile 1-channel to 3-channel
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Resize to 299x299 (required by Inception)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            with torch.no_grad():
                feat = inception(batch)
            
            features_list.append(feat.cpu())
        
        return torch.cat(features_list, dim=0).numpy()
    
    # Extract features
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Compute FID
    fid = _compute_fid_from_stats(mu_real, sigma_real, mu_gen, sigma_gen)
    
    print(f"  FID Score: {fid:.2f}")
    return fid


def _compute_fid_from_stats(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID from mean and covariance statistics."""
    from scipy import linalg
    
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical instability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid)


# ─── Power Spectrum Analysis ───────────────────────────────────────────────

def compute_power_spectrum(images):
    """
    Compute radially-averaged 2D power spectrum.
    
    This is how astrophysicists validate simulated data —
    linking our ML evaluation to physics methodology.
    
    Args:
        images: numpy array (N, H, W) in [0, 1]
        
    Returns:
        k_bins: Spatial frequency bins
        ps_mean: Mean power spectrum
        ps_std: Standard deviation of power spectrum
    """
    N, H, W = images.shape
    assert H == W, "Images must be square"
    
    power_spectra = []
    
    for i in range(N):
        # 2D FFT
        fft = np.fft.fft2(images[i])
        fft_shifted = np.fft.fftshift(fft)
        power = np.abs(fft_shifted) ** 2
        
        # Radial averaging
        center = H // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype(int)
        
        max_r = min(center, W // 2)
        radial_ps = np.zeros(max_r)
        
        for ri in range(max_r):
            mask = (r == ri)
            if mask.any():
                radial_ps[ri] = power[mask].mean()
        
        power_spectra.append(radial_ps)
    
    power_spectra = np.array(power_spectra)
    k_bins = np.arange(power_spectra.shape[1])
    
    return k_bins, power_spectra.mean(axis=0), power_spectra.std(axis=0)


def visualize_power_spectrum(
    real_images, 
    ddpm_images, 
    vae_images=None,
    save_path=None
):
    """
    Compare power spectra of real, DDPM-generated, and VAE-generated images.
    
    If DDPM properly captures the spatial structure of lensing images,
    the power spectra should closely match.
    
    Args:
        real_images: (N, H, W) numpy array
        ddpm_images: (N, H, W) numpy array
        vae_images: (N, H, W) numpy array (optional)
        save_path: Path to save the plot
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "power_spectrum.png")
    
    print("Computing power spectra...")
    
    k_real, ps_real, std_real = compute_power_spectrum(real_images)
    k_ddpm, ps_ddpm, std_ddpm = compute_power_spectrum(ddpm_images)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot with confidence bands
    ax.semilogy(k_real, ps_real, 'b-', linewidth=2, label='Real')
    ax.fill_between(k_real, ps_real - std_real, ps_real + std_real, 
                    alpha=0.2, color='blue')
    
    ax.semilogy(k_ddpm, ps_ddpm, 'r--', linewidth=2, label='DDPM')
    ax.fill_between(k_ddpm, ps_ddpm - std_ddpm, ps_ddpm + std_ddpm, 
                    alpha=0.2, color='red')
    
    if vae_images is not None:
        k_vae, ps_vae, std_vae = compute_power_spectrum(vae_images)
        ax.semilogy(k_vae, ps_vae, 'g-.', linewidth=2, label='VAE')
        ax.fill_between(k_vae, ps_vae - std_vae, ps_vae + std_vae, 
                        alpha=0.2, color='green')
    
    ax.set_xlabel('Spatial Frequency k', fontsize=12)
    ax.set_ylabel('Power Spectrum P(k)', fontsize=12)
    ax.set_title('Radially-Averaged Power Spectrum: Real vs Generated', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Power spectrum saved to {save_path}")


def visualize_pixel_distributions(
    real_images, 
    ddpm_images, 
    vae_images=None,
    save_path=None
):
    """
    Compare pixel intensity distributions.
    
    Args:
        real_images: (N, H, W) numpy array in [0, 1]
        ddpm_images: (N, H, W) numpy array in [0, 1]
        vae_images: (N, H, W) numpy array (optional)
    """
    save_path = save_path or os.path.join(config.RESULTS_DIR, "pixel_distributions.png")
    
    print("Plotting pixel distributions...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(real_images.flatten(), bins=100, alpha=0.6, density=True, 
            color='blue', label='Real')
    ax.hist(ddpm_images.flatten(), bins=100, alpha=0.6, density=True, 
            color='red', label='DDPM')
    
    if vae_images is not None:
        ax.hist(vae_images.flatten(), bins=100, alpha=0.5, density=True, 
                color='green', label='VAE')
    
    ax.set_xlabel('Pixel Intensity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Pixel Intensity Distribution: Real vs Generated', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pixel distributions saved to {save_path}")


def full_evaluation(
    ddpm_model,
    train_loader,
    vae_model=None,
    diffusion=None,
    n_samples=None
):
    """
    Run full evaluation: FID + power spectrum + pixel distributions.
    
    Args:
        ddpm_model: Trained DDPM model
        train_loader: DataLoader with real images
        vae_model: Optional trained VAE
        diffusion: GaussianDiffusion instance
        n_samples: Number of samples for FID
        
    Returns:
        results: dict with FID scores
    """
    from .diffusion import GaussianDiffusion
    
    diffusion = diffusion or GaussianDiffusion()
    n_samples = n_samples or config.FID_NUM_SAMPLES
    
    results = {}
    
    # Collect real images
    print(f"\nCollecting {n_samples} real images...")
    real_list = []
    for batch in train_loader:
        real_list.append(batch)
        if sum(b.shape[0] for b in real_list) >= n_samples:
            break
    real_images = torch.cat(real_list, dim=0)[:n_samples]
    real_images_01 = (real_images.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]
    
    # Generate DDPM samples
    print(f"Generating {n_samples} DDPM samples...")
    ddpm_model.eval()
    ddpm_list = []
    batch_gen = 32
    for i in tqdm(range(0, n_samples, batch_gen)):
        n = min(batch_gen, n_samples - i)
        with torch.no_grad():
            samples = diffusion.sample(ddpm_model, n_samples=n)
        ddpm_list.append(samples.cpu())
    ddpm_images = torch.cat(ddpm_list, dim=0)
    ddpm_images_01 = (ddpm_images.clamp(-1, 1) + 1) / 2
    
    # FID: DDPM
    results['fid_ddpm'] = calculate_fid(real_images_01, ddpm_images_01)
    
    # VAE evaluation (if provided)
    vae_images_01 = None
    if vae_model is not None:
        print(f"Generating {n_samples} VAE samples...")
        vae_model.eval()
        vae_list = []
        for i in range(0, n_samples, batch_gen):
            n = min(batch_gen, n_samples - i)
            with torch.no_grad():
                samples = vae_model.generate(n)
            vae_list.append(samples.cpu())
        vae_images = torch.cat(vae_list, dim=0)
        vae_images_01 = (vae_images.clamp(-1, 1) + 1) / 2
        
        results['fid_vae'] = calculate_fid(real_images_01, vae_images_01)
    
    # Power spectrum analysis
    real_np = real_images_01[:500, 0].numpy()  # Use subset for speed
    ddpm_np = ddpm_images_01[:500, 0].numpy()
    vae_np = vae_images_01[:500, 0].numpy() if vae_images_01 is not None else None
    
    visualize_power_spectrum(real_np, ddpm_np, vae_np)
    
    # Pixel distributions
    visualize_pixel_distributions(real_np, ddpm_np, vae_np)
    
    # Print results summary
    print(f"\n{'='*40}")
    print("EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"  DDPM FID: {results['fid_ddpm']:.2f}")
    if 'fid_vae' in results:
        print(f"  VAE  FID: {results['fid_vae']:.2f}")
    print(f"{'='*40}\n")
    
    return results
