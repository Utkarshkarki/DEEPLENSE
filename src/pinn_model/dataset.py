"""
Dataset loading for 3-class gravitational lensing classification.

Classes: no substructure, subhalo (sphere), vortex (vort)
Includes data augmentation: rotation, flips, Gaussian noise.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F

from . import config


class LensingClassificationDataset(Dataset):
    """
    Dataset for 3-class gravitational lensing images.
    
    Loads .npy files from class-organized directories.
    Applies augmentation during training.
    """
    
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.augment = augment
        
        self.file_paths = []
        self.labels = []
        
        for class_name, label in config.CLASS_LABELS.items():
            class_dir = os.path.join(root_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
            self.file_paths.extend(files)
            self.labels.extend([label] * len(files))
        
        print(f"Loaded {len(self.file_paths)} images from {root_dir}")
        for name, label in config.CLASS_LABELS.items():
            count = self.labels.count(label)
            print(f"  {name}: {count}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load image: (1, 150, 150), float64, [0, 1]
        img = np.load(self.file_paths[idx]).astype(np.float32)
        img = torch.from_numpy(img)  # (1, 150, 150)
        label = self.labels[idx]
        
        # Augmentation
        if self.augment:
            img = self._augment(img)
        
        return img, label
    
    def _augment(self, img):
        """Apply data augmentation."""
        # Random rotation (0, 90, 180, 270 degrees)
        if config.AUG_ROTATION:
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, dims=[1, 2])
        
        # Random horizontal flip
        if config.AUG_FLIP and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])
        
        # Random vertical flip
        if config.AUG_FLIP and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[1])
        
        # Gaussian noise
        if config.AUG_NOISE_STD > 0:
            noise = torch.randn_like(img) * config.AUG_NOISE_STD
            img = torch.clamp(img + noise, 0, 1)
        
        return img


def get_dataloaders(batch_size=None, data_fraction=1.0):
    """
    Create train and validation DataLoaders.
    
    Args:
        batch_size: Override batch size
        data_fraction: Fraction of training data to use (for data efficiency test)
        
    Returns:
        train_loader, val_loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    
    train_dataset = LensingClassificationDataset(config.TRAIN_DIR, augment=True)
    val_dataset = LensingClassificationDataset(config.VAL_DIR, augment=False)
    
    # Data efficiency: use subset of training data
    if data_fraction < 1.0:
        n = int(len(train_dataset) * data_fraction)
        indices = torch.randperm(len(train_dataset))[:n].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"Data efficiency: using {n}/{len(train_dataset)} training samples ({data_fraction*100:.0f}%)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader
