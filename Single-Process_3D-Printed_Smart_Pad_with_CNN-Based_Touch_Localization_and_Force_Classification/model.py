# Imports
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM)
    
    Recalibrates channel-wise features using global average pooling and 
    two 1x1 convolutions to learn channel importance weights.
    
    Input:  x ∈ R^{B, C, W}  (batch, channels, time_steps)
    Output: y ∈ R^{B, C, W}  (recalibrated with residual connection)
    """
    def __init__(self, C: int):
        super().__init__()
        hidden = max(1, C // 2)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Conv1d(C, hidden, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(hidden, C, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.avg_pool(x)            # (B, C, 1)
        u = self.relu(self.conv1(z))    # (B, C/2, 1)
        w = self.sigmoid(self.conv2(u)) # (B, C, 1) - channel weights
        return x * w + x                # residual connection


class TimeAttentionModule(nn.Module):
    """
    Time Attention Module (TAM)
    
    Learns temporal importance using depthwise convolution and 
    generates time-step weights via sigmoid gating.
    
    Input:  x ∈ R^{B, C, W}  (batch, channels, time_steps)
    Output: y ∈ R^{B, C, W}  (temporally weighted with residual)
    """
    def __init__(self, C: int, k_temporal: int = 3):
        super().__init__()
        self.to_time = nn.Conv1d(C, 1, kernel_size=1, bias=True)
        pad = k_temporal // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=k_temporal, padding=pad, groups=C, bias=True),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.to_time(x)          # (B, 1, W)
        aw = self.gate(a)            # (B, 1, W) - time weights
        v = self.temporal(x)         # (B, C, W) - temporal features
        return v * aw + x            # residual connection

    
class Attention1DConv(nn.Module):
    """
    1D Attention CNN for multi-task tactile sensing
    
    Architecture:
    - Input:  (B, T, 4) → transposed to (B, 4, T)
    - Backbone: 7 Conv1D blocks with TAM and CAM attention
    - Location head: 4 fully-connected layers → (x, y) coordinates
    - Force head: 4 fully-connected layers → 3 force classes
    
    Each head has a hardcoded width of 64 neurons per layer.
    """
    def __init__(self):
        super().__init__()

        self.in_channel = 4  # 4 tactile sensor channels
        
        # Backbone: (kernel_size, out_channels, stride)
        backbone = [
            (32, 32, 1),   # Layer 1
            (16, 32, 2),   # Layer 2
            (9, 64, 2),    # Layer 3
            (6, 64, 2),    # Layer 4
            (3, 128, 5),   # Layer 5
            (3, 128, 5),   # Layer 6
            (3, 256, 2)    # Layer 7
        ]

        in_ch = self.in_channel
        layers = []
        for k, out_ch, stride in backbone:
            pad = k // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                TimeAttentionModule(out_ch),
                ChannelAttentionModule(out_ch),
            ]
            in_ch = out_ch
        
        self.out_channel = out_ch  # 256
        self.body = nn.Sequential(*layers)

        # Global pooling to create latent vector
        self.latent_vector = nn.AdaptiveAvgPool1d(1)

        # Location prediction head (4 layers, width=64)
        self.location_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # (x, y) coordinates
        )

        # Force classification head (4 layers, width=64)
        self.force_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # 3 force classes
        )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent features from input time series."""
        x = x.permute(0, 2, 1)         # (B, T, 4) → (B, 4, T)
        y = self.body(x)               # (B, 256, T')
        z = self.latent_vector(y)      # (B, 256, 1)
        return z.squeeze(-1)           # (B, 256)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, T, 4)
            
        Returns:
            loc_logits: Location predictions (B, 2)
            force_logits: Force class logits (B, 3)
        """
        feat = self.forward_features(x)
        loc_logits = self.location_head(feat)
        force_logits = self.force_head(feat)
        return loc_logits, force_logits
