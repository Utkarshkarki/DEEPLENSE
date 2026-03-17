"""
Dataset loading for strong gravitational lensing images.
Loads .npy files, resizes to 128x128, normalizes to [-1, 1].
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from . import config


class LensingDataset(Dataset):
    """
    Custom dataset for gravitational lensing images stored as .npy files.
    
    Each .npy file contains a (1, 150, 150) float64 array in range [0, 1].
    We resize to IMAGE_SIZE and rescale to [-1, 1] for diffusion training.
    """
    
    def __init__(self, data_dir=None, image_size=None):
        self.data_dir = data_dir or config.DATA_DIR
        self.image_size = image_size or config.IMAGE_SIZE
        
        # Find all .npy files
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.data_dir}. "
                "Please check the dataset path."
            )
        
        print(f"Found {len(self.file_paths)} lensing images in {self.data_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load image: shape (1, 150, 150), dtype float64, range [0, 1]
        img = np.load(self.file_paths[idx]).astype(np.float32)
        img = torch.from_numpy(img)  # (1, 150, 150)
        
        # Resize to target size
        if img.shape[-1] != self.image_size:
            img = img.unsqueeze(0)  # (1, 1, 150, 150) for F.interpolate
            img = F.interpolate(img, size=self.image_size, mode='bilinear', align_corners=False)
            img = img.squeeze(0)    # (1, 128, 128)
        
        # Rescale [0, 1] → [-1, 1]
        img = img * 2.0 - 1.0
        
        return img


def get_dataloaders(batch_size=None, train_split=None):
    """
    Create train and validation DataLoaders.
    
    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    train_split = train_split or config.TRAIN_SPLIT
    
    dataset = LensingDataset()
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train: {train_size} images, Val: {val_size} images")
    print(f"Batch size: {batch_size}, Train batches: {len(train_loader)}")
    
    return train_loader, val_loader
