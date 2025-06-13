"""
NumPy-based unified renderer with Numba acceleration.
"""

import torch
import numpy as np
from numba import jit
from typing import Tuple

from .base_renderer import BaseUnifiedRenderer


class NumPyRenderer(BaseUnifiedRenderer):
    """NumPy/Numba renderer for CPU execution comparison."""
    
    def __init__(self):
        super().__init__("NumPy + Numba", "cpu")
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """NumPy-based volume rendering with Numba acceleration."""
        # Convert to numpy for Numba processing
        densities_np = densities.cpu().numpy()
        colors_np = colors.cpu().numpy()
        z_vals_np = z_vals.cpu().numpy()
        ray_dirs_np = ray_directions.cpu().numpy()
        
        # Use compiled Numba functions
        rgb_np, depth_np = self._numba_volume_render(
            densities_np, colors_np, z_vals_np, ray_dirs_np
        )
        
        # Convert back to torch
        rgb_map = torch.from_numpy(rgb_np).float()
        depth_map = torch.from_numpy(depth_np).float()
        
        return rgb_map, depth_map
    
    @staticmethod
    @jit(nopython=True, parallel=False)
    def _numba_volume_render(densities, colors, z_vals, ray_directions):
        """Numba-compiled volume rendering for maximum CPU performance (no parallel due to macOS OpenMP issues)."""
        n_rays, n_samples = z_vals.shape
        rgb_map = np.zeros((n_rays, 3), dtype=np.float32)
        depth_map = np.zeros(n_rays, dtype=np.float32)
        
        for ray_idx in range(n_rays):
            # Calculate distances
            dists = np.zeros(n_samples, dtype=np.float32)
            for i in range(n_samples - 1):
                dists[i] = z_vals[ray_idx, i + 1] - z_vals[ray_idx, i]
            dists[-1] = 1e10
            
            # Apply ray direction magnitude
            ray_norm = np.sqrt(ray_directions[ray_idx, 0]**2 + 
                             ray_directions[ray_idx, 1]**2 + 
                             ray_directions[ray_idx, 2]**2)
            dists *= ray_norm
            
            # Volume rendering integration
            transmittance = 1.0
            for i in range(n_samples):
                density = max(0.0, densities[ray_idx, i, 0])
                alpha = 1.0 - np.exp(-density * dists[i])
                
                weight = transmittance * alpha
                
                # Accumulate color and depth
                for c in range(3):
                    rgb_map[ray_idx, c] += weight * colors[ray_idx, i, c]
                depth_map[ray_idx] += weight * z_vals[ray_idx, i]
                
                transmittance *= (1.0 - alpha)
                
                # Early termination for efficiency
                if transmittance < 0.01:
                    break
        
        return rgb_map, depth_map
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """NumPy-based image rendering."""
        width, height = resolution
        
        # Generate rays using NumPy for consistency
        rays_o, rays_d = self._generate_rays_numpy(camera_pose.cpu().numpy(), width, height)
        
        # Process in chunks
        chunk_size = 256  # Smaller chunks for CPU/NumPy
        rgb_chunks = []
        depth_chunks = []
        
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        for i in range(0, rays_o_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, rays_o_flat.shape[0])
            
            rgb_chunk, depth_chunk = self._render_ray_chunk_numpy(
                rays_o_flat[i:end_idx], rays_d_flat[i:end_idx], samples_per_ray
            )
            
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
        
        rgb_img = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3)
        depth_img = torch.cat(depth_chunks, dim=0).reshape(height, width)
        
        return rgb_img, depth_img
    
    def _generate_rays_numpy(self, c2w: np.ndarray, width: int, height: int):
        """NumPy ray generation for consistency."""
        focal = 800.0
        
        i, j = np.meshgrid(np.linspace(0, width-1, width), 
                          np.linspace(0, height-1, height), indexing='ij')
        i = i.T
        j = j.T
        
        dirs = np.stack([
            (i - width * 0.5) / focal,
            -(j - height * 0.5) / focal,
            -np.ones_like(i)
        ], axis=-1)
        
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        
        # Make arrays writable to avoid PyTorch warnings
        rays_o = rays_o.copy()
        rays_d = rays_d.copy()
        
        return torch.from_numpy(rays_o).float(), torch.from_numpy(rays_d).float()
    
    def _render_ray_chunk_numpy(self, rays_o: np.ndarray, rays_d: np.ndarray, samples_per_ray: int):
        """Render ray chunk with NumPy integration but NeRF network queries."""
        # Sample points (still use PyTorch for consistency)
        rays_o_torch = torch.from_numpy(rays_o).float() if isinstance(rays_o, np.ndarray) else rays_o
        rays_d_torch = torch.from_numpy(rays_d).float() if isinstance(rays_d, np.ndarray) else rays_d
        
        points, z_vals = self.sample_points_on_rays(rays_o_torch, rays_d_torch, samples_per_ray)
        
        # Flatten for NeRF query
        points_flat = points.reshape(-1, 3)
        dirs_flat = rays_d_torch[:, None, :].expand_as(points).reshape(-1, 3)
        
        # Query NeRF network (still uses shared PyTorch model)
        densities, colors = self.query_nerf_networks(points_flat, dirs_flat, use_fine=True)
        
        # Reshape and use NumPy volume rendering
        densities = densities.reshape(*points.shape[:-1], 1)
        colors = colors.reshape(*points.shape)
        
        rgb, depth = self.execute_volume_rendering(densities, colors, z_vals, rays_d_torch)
        
        return rgb, depth
