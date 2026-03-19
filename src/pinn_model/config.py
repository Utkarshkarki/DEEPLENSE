"""
Configuration for Consistency-Regularized Lensiformer.
"""

import os
import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "test7", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "pinn")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "pinn")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 150          # Native resolution (no resize needed)
CHANNELS = 1              # Grayscale lensing images
NUM_CLASSES = 3
CLASS_NAMES = ["no", "sphere", "vort"]
CLASS_LABELS = {"no": 0, "sphere": 1, "vort": 2}

# ─── Augmentation ───────────────────────────────────────────────────────────
AUG_ROTATION = True       # Random rotation (critical for lensing symmetry)
AUG_FLIP = True           # Horizontal + vertical flips
AUG_NOISE_STD = 0.01      # Gaussian noise σ

# ─── Physics Encoder ────────────────────────────────────────────────────────
K_SIS_INIT = 1.0          # Initial SIS k value
K_RANGE = 0.2             # k saturated to ±20% of SIS
PSI_RES_SCALE = 0.1       # ψ_residual magnitude cap
GAUSSIAN_SIGMA = 1.0      # Gaussian blur σ for ψ smoothing
IDENTITY_SKIP_WEIGHT = 0.1  # S = (1-w)*warped + w*I

# ─── ViT / Transformer ─────────────────────────────────────────────────────
PATCH_SIZE = 15            # 150/15 = 10 patches per side = 100 tokens
VIT_DIM = 256              # Token embedding dimension
VIT_DEPTH = 6              # Transformer blocks
VIT_HEADS = 8              # Attention heads
VIT_MLP_DIM = 512          # MLP hidden dim in transformer
VIT_DROPOUT = 0.1
MC_DROPOUT_RATE = 0.1      # MC Dropout for uncertainty
MC_SAMPLES = 10            # Number of forward passes for uncertainty

# ─── CNN Backbone (hybrid) ──────────────────────────────────────────────────
CNN_FEATURES = 128         # CNN output features before ViT

# ─── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5          # LR warmup
SAVE_INTERVAL = 10

# ─── Loss Weights ───────────────────────────────────────────────────────────
LAMBDA_PHYSICS = 0.1       # L_physics (reconstruction consistency)
LAMBDA_K_SMOOTH = 0.01     # L_k (k-map smoothness)
LAMBDA_RES = 0.01          # L_res (ψ_residual penalty)
LAMBDA_ALPHA = 0.001       # L_alpha (deflection energy)

# ─── Curriculum Learning ────────────────────────────────────────────────────
CURRICULUM_START = 10      # Epoch to start physics loss
CURRICULUM_END = 50        # Epoch when physics loss reaches full weight

# ─── Evaluation ─────────────────────────────────────────────────────────────
DATA_EFFICIENCY_FRACTIONS = [0.1, 0.3, 1.0]  # For data efficiency test
