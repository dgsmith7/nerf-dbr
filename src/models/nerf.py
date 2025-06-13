"""
NeRF Model Implementation

A clean implementation of Neural Radiance Fields optimized for M3 Pro.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding:
    """Positional encoding for NeRF inputs."""
    
    def __init__(self, L: int = 10):
        """
        Args:
            L: Number of frequency bands for encoding
        """
        self.L = L
        self.freq_bands = 2.0 ** torch.linspace(0, L-1, L)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input tensor of shape [..., 3]
            
        Returns:
            Encoded tensor of shape [..., 3 + 3*2*L]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Move frequency bands to same device as input
        self.freq_bands = self.freq_bands.to(x.device)
        
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * torch.pi * x))
            out.append(torch.cos(freq * torch.pi * x))
        
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    """Neural Radiance Field model."""
    
    def __init__(self, pos_L: int = 10, dir_L: int = 4, hidden_dim: int = 256):
        """
        Args:
            pos_L: Position encoding frequency bands
            dir_L: Direction encoding frequency bands  
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.pos_L = pos_L
        self.dir_L = dir_L
        
        # Input dimensions after positional encoding
        self.pos_dim = 3 + 3 * 2 * pos_L  # 60D for L=10
        self.dir_dim = 3 + 3 * 2 * dir_L  # 24D for L=4
        
        # Positional encoders
        self.pos_encoder = PositionalEncoding(pos_L)
        self.dir_encoder = PositionalEncoding(dir_L)
        
        # Main MLP layers (1-8)
        self.layers = nn.ModuleList([
            nn.Linear(self.pos_dim, hidden_dim),                    # Layer 1
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 2
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 3
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 4
            nn.Linear(hidden_dim + self.pos_dim, hidden_dim),       # Layer 5 (skip)
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 6
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 7
            nn.Linear(hidden_dim, hidden_dim),                      # Layer 8
        ])
        
        # Output heads
        self.density_head = nn.Linear(hidden_dim, 1)
        
        # Color head
        self.color_layers = nn.ModuleList([
            nn.Linear(hidden_dim + self.dir_dim, hidden_dim // 2),  # Layer 9
            nn.Linear(hidden_dim // 2, 3)                           # Layer 10
        ])
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor = None) -> tuple:
        """
        Forward pass through NeRF network.
        
        Args:
            positions: 3D positions [N, 3]
            directions: Viewing directions [N, 3]
            
        Returns:
            (density, rgb): Volume density [N, 1] and RGB color [N, 3]
        """
        # Encode positions
        pos_encoded = self.pos_encoder.encode(positions)
        
        # Main MLP forward pass
        x = pos_encoded
        for i, layer in enumerate(self.layers):
            if i == 4:  # Skip connection at layer 5
                x = torch.cat([x, pos_encoded], dim=-1)
            x = F.relu(layer(x))
        
        # Density prediction
        density = F.relu(self.density_head(x))
        
        # Color prediction
        if directions is not None:
            dir_encoded = self.dir_encoder.encode(directions)
            color_input = torch.cat([x, dir_encoded], dim=-1)
        else:
            color_input = x
        
        color = color_input
        for i, layer in enumerate(self.color_layers):
            color = layer(color)
            if i < len(self.color_layers) - 1:
                color = F.relu(color)
        
        color = torch.sigmoid(color)
        
        return density, color
