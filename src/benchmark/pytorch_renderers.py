"""
PyTorch-based unified renderers for different devices.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from .base_renderer import BaseUnifiedRenderer


class PyTorchMPSRenderer(BaseUnifiedRenderer):
    """PyTorch MPS (Metal Performance Shaders) renderer for M3 Pro."""
    
    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this system")
        
        super().__init__("PyTorch MPS", "mps")
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MPS-optimized volume rendering."""
        # Distance between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
        
        # Alpha compositing with MPS optimization
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                 transmittance[..., :-1]], dim=-1)
        
        weights = alpha * transmittance
        
        # Composite colors and depth
        rgb_map = torch.sum(weights[..., None] * colors, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        return rgb_map, depth_map
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render full image using MPS acceleration."""
        width, height = resolution
        
        # Generate rays
        rays_o, rays_d = self.generate_rays(camera_pose, width, height)
        
        # Flatten for batch processing
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # Process in chunks optimized for MPS
        chunk_size = 2048  # Larger chunks for MPS efficiency
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, rays_o_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, rays_o_flat.shape[0])
            
            rgb_chunk, depth_chunk = self._render_ray_chunk(
                rays_o_flat[i:end_idx], rays_d_flat[i:end_idx], samples_per_ray
            )
            
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
        
        # Combine chunks and reshape
        rgb_img = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth_img = torch.cat(depth_chunks, dim=0).reshape(height, width)
        
        return rgb_img, depth_img
    
    def _render_ray_chunk(self, rays_o: torch.Tensor, rays_d: torch.Tensor, samples_per_ray: int):
        """Render a chunk of rays using MPS."""
        # Sample points along rays
        points, z_vals = self.sample_points_on_rays(rays_o, rays_d, samples_per_ray)
        
        # Flatten for network query
        points_flat = points.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand_as(points).reshape(-1, 3)
        
        # Query NeRF network (using fine network)
        densities, colors = self.query_nerf_networks(points_flat, dirs_flat, use_fine=True)
        
        # Reshape back
        densities = densities.reshape(*points.shape[:-1], 1)
        colors = colors.reshape(*points.shape)
        
        # Volume render
        rgb, depth = self.execute_volume_rendering(densities, colors, z_vals, rays_d)
        
        return rgb, depth


class PyTorchCPURenderer(BaseUnifiedRenderer):
    """PyTorch CPU renderer with optimizations."""
    
    def __init__(self):
        super().__init__("PyTorch CPU", "cpu")
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU-optimized volume rendering with correct transmittance formula."""
        # Distance between samples (same as MPS renderer)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
        
        # Alpha compositing (same as MPS renderer)
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                 transmittance[..., :-1]], dim=-1)
        
        weights = alpha * transmittance
        
        # Composite colors and depth
        rgb_map = torch.sum(weights[..., None] * colors, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        return rgb_map, depth_map
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """CPU-optimized image rendering."""
        width, height = resolution
        
        rays_o, rays_d = self.generate_rays(camera_pose, width, height)
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # Smaller chunks for CPU to manage memory
        chunk_size = 512
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, rays_o_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, rays_o_flat.shape[0])
            
            rgb_chunk, depth_chunk = self._render_ray_chunk(
                rays_o_flat[i:end_idx], rays_d_flat[i:end_idx], samples_per_ray
            )
            
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
        
        rgb_img = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth_img = torch.cat(depth_chunks, dim=0).reshape(height, width)
        
        return rgb_img, depth_img
    
    def _render_ray_chunk(self, rays_o: torch.Tensor, rays_d: torch.Tensor, samples_per_ray: int):
        """Render ray chunk with CPU optimizations."""
        points, z_vals = self.sample_points_on_rays(rays_o, rays_d, samples_per_ray)
        
        points_flat = points.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand_as(points).reshape(-1, 3)
        
        densities, colors = self.query_nerf_networks(points_flat, dirs_flat, use_fine=True)
        
        densities = densities.reshape(*points.shape[:-1], 1)
        colors = colors.reshape(*points.shape)
        
        rgb, depth = self.execute_volume_rendering(densities, colors, z_vals, rays_d)
        
        return rgb, depth


class PyTorchCUDARenderer(BaseUnifiedRenderer):
    """PyTorch CUDA renderer (if available)."""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this system")
        
        super().__init__("PyTorch CUDA", "cuda")
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CUDA-optimized volume rendering."""
        # CUDA-optimized operations
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
        
        # Fused operations for CUDA efficiency
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                 transmittance[..., :-1]], dim=-1)
        
        weights = alpha * transmittance
        
        rgb_map = torch.sum(weights[..., None] * colors, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        return rgb_map, depth_map
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """CUDA-optimized image rendering."""
        width, height = resolution
        
        rays_o, rays_d = self.generate_rays(camera_pose, width, height)
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # Larger chunks for CUDA
        chunk_size = 4096
        rgb_chunks = []
        depth_chunks = []
        
        for i in range(0, rays_o_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, rays_o_flat.shape[0])
            
            rgb_chunk, depth_chunk = self._render_ray_chunk(
                rays_o_flat[i:end_idx], rays_d_flat[i:end_idx], samples_per_ray
            )
            
            rgb_chunks.append(rgb_chunk.cpu())  # Move to CPU for storage
            depth_chunks.append(depth_chunk.cpu())
        
        rgb_img = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth_img = torch.cat(depth_chunks, dim=0).reshape(height, width)
        
        return rgb_img, depth_img
    
    def _render_ray_chunk(self, rays_o: torch.Tensor, rays_d: torch.Tensor, samples_per_ray: int):
        """CUDA-optimized ray chunk rendering."""
        points, z_vals = self.sample_points_on_rays(rays_o, rays_d, samples_per_ray)
        
        points_flat = points.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand_as(points).reshape(-1, 3)
        
        densities, colors = self.query_nerf_networks(points_flat, dirs_flat, use_fine=True)
        
        densities = densities.reshape(*points.shape[:-1], 1)
        colors = colors.reshape(*points.shape)
        
        rgb, depth = self.execute_volume_rendering(densities, colors, z_vals, rays_d)
        
        return rgb, depth
