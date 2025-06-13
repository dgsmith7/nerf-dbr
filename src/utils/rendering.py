"""
Volume rendering utilities for NeRF.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class VolumeRenderer:
    """Volume rendering implementation for NeRF."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def sample_points_on_rays(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor,
                             near: float = 2.0, far: float = 6.0, n_samples: int = 64,
                             perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays for volume rendering.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            near: Near plane distance
            far: Far plane distance
            n_samples: Number of samples per ray
            perturb: Whether to perturb sampling locations
            
        Returns:
            (points, z_vals): 3D points [N, n_samples, 3] and depth values [N, n_samples]
        """
        batch_size = ray_origins.shape[0]
        
        # Linear sampling in depth
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=self.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = z_vals.expand(batch_size, n_samples)
        
        # Perturb sampling locations for training
        if perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
        
        # Get 3D points along rays
        points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
        
        return points, z_vals
    
    def importance_sample(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor,
                         z_vals: torch.Tensor, weights: torch.Tensor,
                         n_importance: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Importance sampling for fine network.
        
        Args:
            ray_origins: Ray origins [N, 3]
            ray_directions: Ray directions [N, 3]
            z_vals: Coarse depth values [N, n_coarse]
            weights: Weights from coarse network [N, n_coarse]
            n_importance: Number of importance samples
            
        Returns:
            (points, z_vals): Fine sample points and depth values
        """
        batch_size = ray_origins.shape[0]
        
        # Get pdf from weights
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Sample from CDF
        u = torch.rand(batch_size, n_importance, device=self.device)
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, z_vals.shape[-1] - 1)
        above = torch.clamp(indices, 0, z_vals.shape[-1] - 1)
        
        # Linear interpolation
        indices_g = torch.stack([below, above], dim=-1)
        matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, 
                           index=indices_g)
        z_vals_g = torch.gather(z_vals.unsqueeze(-2).expand(matched_shape), dim=-1,
                              index=indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        z_samples = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
        
        # Get 3D points
        points = ray_origins[..., None, :] + ray_directions[..., None, :] * z_samples[..., :, None]
        
        return points, z_samples
    
    def volume_render(self, densities: torch.Tensor, colors: torch.Tensor,
                     z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform volume rendering using densities and colors.
        
        Args:
            densities: Volume densities [N, n_samples, 1]
            colors: RGB colors [N, n_samples, 3]
            z_vals: Depth values [N, n_samples]
            ray_directions: Ray directions [N, 3]
            
        Returns:
            (rgb_map, depth_map, acc_map, weights): Rendered outputs
        """
        # Distance between adjacent samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Multiply each distance by the norm of its corresponding direction ray
        dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                 transmittance[..., :-1]], dim=-1)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Composite colors
        rgb_map = torch.sum(weights[..., None] * colors, dim=-2)
        
        # Compute depth map
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        # Compute accumulated alpha (opacity)
        acc_map = torch.sum(weights, dim=-1)
        
        return rgb_map, depth_map, acc_map, weights
