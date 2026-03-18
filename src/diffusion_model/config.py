"""
Configuration for DDPM Gravitational Lensing Image Generation.
All hyperparameters and paths are defined here for easy experimentation.
"""

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "Samples", "Samples")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "diffusion")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "diffusion")

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 128          # Resize from 150x150 to 128x128 (power-of-2)
CHANNELS = 1              # Grayscale lensing images
TRAIN_SPLIT = 0.9         # 90% train, 10% validation

# ─── Diffusion Process ──────────────────────────────────────────────────────
TIMESTEPS = 1000          # Number of diffusion steps
SCHEDULE_TYPE = "cosine"  # "cosine" (Nichol & Dhariwal 2021) or "linear"
# Linear schedule params (used if SCHEDULE_TYPE == "linear")
BETA_START = 1e-4
BETA_END = 0.02
# Cosine schedule params
COSINE_S = 0.008          # Small offset to prevent β near 0

# ─── U-Net Architecture ─────────────────────────────────────────────────────
BASE_CHANNELS = 32
CHANNEL_MULTS = (1, 2, 4, 4)   # → 32, 64, 128, 128 (optimized for 4GB GPU)
ATTENTION_RESOLUTIONS = (16,)    # Attention at 16x16 only (memory-efficient)
NUM_RES_BLOCKS = 2
TIME_EMB_DIM = 128
DROPOUT = 0.1

# ─── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 8                # Small batch for 4GB GPU
LEARNING_RATE = 2e-4
NUM_EPOCHS = 200
EMA_DECAY = 0.9999
SAVE_INTERVAL = 20        # Save checkpoint every N epochs
SAMPLE_INTERVAL = 10      # Generate samples every N epochs

# ─── VAE Baseline ───────────────────────────────────────────────────────────
VAE_LATENT_DIM = 128
VAE_EPOCHS = 100
VAE_LR = 1e-3
VAE_KL_WEIGHT = 0.001     # Weight for KL divergence loss

# ─── Ablation ───────────────────────────────────────────────────────────────
ABLATION_EPOCHS = 50      # Shorter training for ablation experiments

# ─── Evaluation ─────────────────────────────────────────────────────────────
FID_NUM_SAMPLES = 2000    # Number of generated samples for FID
FID_BATCH_SIZE = 16           # Small batch for 4GB GPU

# ─── Interpolation ──────────────────────────────────────────────────────────
INTERP_STEPS = 10         # Number of interpolation steps between two samples
