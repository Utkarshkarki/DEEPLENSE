"""
Baseline models for comparison with Lensiformer.

1. ResNet-18 (CNN only, no physics)
2. PhysicsCNN (physics encoder + CNN, no ViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .lens_layer import InverseLensLayer


class ResNet18Baseline(nn.Module):
    """
    ResNet-18 adapted for 1-channel 150×150 grayscale lensing images.
    No physics, no transformer — pure CNN baseline.
    """
    
    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        
        self.features = nn.Sequential(
            # Block 1: (1, 150, 150) → (64, 75, 75)
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),  # → (64, 38, 38)
            
            # Block 2
            self._make_block(64, 64),
            self._make_block(64, 64),
            
            # Block 3
            self._make_block(64, 128, stride=2),   # → (128, 19, 19)
            self._make_block(128, 128),
            
            # Block 4
            self._make_block(128, 256, stride=2),   # → (256, 10, 10)
            self._make_block(256, 256),
            
            # Block 5
            self._make_block(256, 512, stride=2),   # → (512, 5, 5)
            self._make_block(512, 512),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def _make_block(self, in_ch, out_ch, stride=1):
        return ResBlock(in_ch, out_ch, stride)
    
    def forward(self, x, return_physics=False):
        feat = self.features(x)
        feat = feat.flatten(1)
        logits = self.classifier(feat)
        
        if return_physics:
            return logits, {}
        return logits


class ResBlock(nn.Module):
    """Residual block for ResNet baseline."""
    
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class PhysicsCNN(nn.Module):
    """
    Physics encoder + CNN (no ViT).
    
    Isolates benefit of physics from transformer.
    Uses inverse lens to reconstruct source, then classifies
    concatenation of source + original with a CNN.
    """
    
    def __init__(self, num_classes=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        
        # Physics encoder
        self.inverse_lens = InverseLensLayer()
        
        # CNN on concatenated [source, original]
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),    # 2 channels: source + original
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),                     # 75
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),                     # 37
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),                     # 18
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, return_physics=False):
        # Physics: reconstruct source
        source, physics_data = self.inverse_lens(image)
        
        # Concat source + original
        combined = torch.cat([source, image], dim=1)  # (B, 2, H, W)
        
        feat = self.features(combined)
        feat = feat.flatten(1)
        logits = self.classifier(feat)
        
        if return_physics:
            return logits, physics_data
        return logits
