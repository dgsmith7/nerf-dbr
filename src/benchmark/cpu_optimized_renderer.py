"""
CPU-optimized ray marching renderer using pure NumPy operations.

This renderer implements the same NeRF algorithm but optimized for pure CPU execution
without PyTorch overhead. It uses the same trained network weights but executes
everything in NumPy for maximum CPU efficiency.
"""

import torch
import numpy as np
from typing import Tuple
import time

from .base_renderer import BaseUnifiedRenderer


class CPUOptimizedRenderer(BaseUnifiedRenderer):
    """Pure CPU implementation using trained MLP weights in NumPy."""
    
    def __init__(self):
        super().__init__("CPU Optimized", "cpu")
        
        # NumPy versions of network weights
        self.coarse_weights = None
        self.fine_weights = None
        
        # Positional encoding parameters
        self.L_pos = 10  # Position encoding levels
        self.L_dir = 4   # Direction encoding levels
        
    def setup(self, checkpoint_path: str):
        """Setup renderer with trained model weights converted to NumPy."""
        super().setup(checkpoint_path)
        
        # Get PyTorch models
        coarse_model, fine_model = self.shared_model.get_models(self.device)
        
        # Convert PyTorch weights to NumPy for CPU optimization
        print("Converting PyTorch weights to optimized NumPy format...")
        self.coarse_weights = self._extract_numpy_weights(coarse_model)
        self.fine_weights = self._extract_numpy_weights(fine_model)
        
    def _extract_numpy_weights(self, model):
        """Extract and convert PyTorch model weights to NumPy format."""
        weights = {}
        state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            weights[name] = param.detach().cpu().numpy()
            
        return weights
    
    def _positional_encoding_numpy(self, x: np.ndarray, L: int) -> np.ndarray:
        """NumPy implementation of positional encoding."""
        batch_size = x.shape[0]
        input_dim = x.shape[1]
        
        # Initialize output array
        output_dim = input_dim * (2 * L + 1)
        encoded = np.zeros((batch_size, output_dim), dtype=np.float32)
        
        # Original coordinates
        encoded[:, :input_dim] = x
        
        # Sinusoidal encodings
        for i in range(L):
            freq = 2.0 ** i
            start_idx = input_dim + i * input_dim * 2
            
            # sin(2^i * pi * x)
            encoded[:, start_idx:start_idx + input_dim] = np.sin(freq * np.pi * x)
            # cos(2^i * pi * x)  
            encoded[:, start_idx + input_dim:start_idx + 2 * input_dim] = np.cos(freq * np.pi * x)
            
        return encoded
    
    def _mlp_forward_numpy(self, positions: np.ndarray, directions: np.ndarray, weights: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through MLP using NumPy operations."""
        
        def relu(x):
            return np.maximum(0, x)
        
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
        def linear_layer(x, weight_name, bias_name):
            weight = weights[weight_name]
            bias = weights[bias_name]
            return np.dot(x, weight.T) + bias
        
        # Encode inputs
        pos_encoded = self._positional_encoding_numpy(positions, self.L_pos)
        dir_encoded = self._positional_encoding_numpy(directions, self.L_dir)
        
        # Forward pass through the network
        x = pos_encoded
        x1 = relu(linear_layer(x, 'layers.0.weight', 'layers.0.bias'))
        x2 = relu(linear_layer(x1, 'layers.1.weight', 'layers.1.bias'))
        x3 = relu(linear_layer(x2, 'layers.2.weight', 'layers.2.bias'))
        x4 = relu(linear_layer(x3, 'layers.3.weight', 'layers.3.bias'))
        
        # Skip connection at layer 4
        x4_skip = np.concatenate([x4, pos_encoded], axis=1)
        
        x5 = relu(linear_layer(x4_skip, 'layers.4.weight', 'layers.4.bias'))
        x6 = relu(linear_layer(x5, 'layers.5.weight', 'layers.5.bias'))
        x7 = relu(linear_layer(x6, 'layers.6.weight', 'layers.6.bias'))
        x8 = relu(linear_layer(x7, 'layers.7.weight', 'layers.7.bias'))
        
        # Density output
        density = relu(linear_layer(x8, 'density_head.weight', 'density_head.bias'))
        
        # Color branch
        color_input = np.concatenate([x8, dir_encoded], axis=1)
        color_hidden = relu(linear_layer(color_input, 'color_layers.0.weight', 'color_layers.0.bias'))
        color = sigmoid(linear_layer(color_hidden, 'color_layers.1.weight', 'color_layers.1.bias'))
        
        return density, color
    
    def query_nerf_networks(self, positions: torch.Tensor, directions: torch.Tensor,
                           use_fine: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query NeRF networks using optimized NumPy operations."""
        
        # Convert to NumPy
        pos_np = positions.detach().cpu().numpy()
        dir_np = directions.detach().cpu().numpy()
        
        # Query coarse network
        coarse_density, coarse_color = self._mlp_forward_numpy(pos_np, dir_np, self.coarse_weights)
        
        # Query fine network
        fine_density, fine_color = self._mlp_forward_numpy(pos_np, dir_np, self.fine_weights)
        
        # Use fine network results if requested
        if use_fine:
            density = fine_density
            color = fine_color
        else:
            density = coarse_density
            color = coarse_color
        
        # Convert back to PyTorch tensors
        density_tensor = torch.from_numpy(density).float()
        color_tensor = torch.from_numpy(color).float()
        
        return density_tensor, color_tensor
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized CPU volume rendering using NumPy."""
        
        # Convert to NumPy for CPU optimization
        densities_np = densities.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()
        z_vals_np = z_vals.detach().cpu().numpy()
        
        # Optimized volume rendering in NumPy
        n_rays, n_samples = z_vals_np.shape
        
        # Calculate distances between samples
        dists = np.zeros_like(z_vals_np)
        dists[:, :-1] = z_vals_np[:, 1:] - z_vals_np[:, :-1]
        dists[:, -1] = 1e10  # Large value for last sample
        
        # Alpha compositing
        alpha = 1.0 - np.exp(-densities_np.squeeze() * dists)
        
        # Cumulative transmittance
        transmittance = np.cumprod(1.0 - alpha + 1e-10, axis=1)
        transmittance = np.concatenate([np.ones((n_rays, 1)), transmittance[:, :-1]], axis=1)
        
        # Weights for each sample
        weights = alpha * transmittance
        
        # Render RGB
        rgb_map = np.sum(weights[..., np.newaxis] * colors_np, axis=1)
        
        # Render depth
        depth_map = np.sum(weights * z_vals_np, axis=1)
        
        # Convert back to PyTorch
        rgb_tensor = torch.from_numpy(rgb_map).float()
        depth_tensor = torch.from_numpy(depth_map).float()
        
        return rgb_tensor, depth_tensor
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render full image using CPU optimized ray marching."""
        width, height = resolution
        
        with self.performance_monitor():
            # Generate rays
            rays_o, rays_d = self.generate_rays(camera_pose, width, height)
            
            # Flatten rays
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            
            # Sample points along rays
            points, z_vals = self.sample_points_on_rays(rays_o, rays_d, samples_per_ray)
            
            # Query NeRF networks
            densities, colors = self.query_nerf_networks(
                points.reshape(-1, 3), 
                rays_d.unsqueeze(1).expand(-1, samples_per_ray, -1).reshape(-1, 3),
                samples_per_ray
            )
            
            # Reshape for volume rendering
            densities = densities.reshape(rays_o.shape[0], samples_per_ray, 1)
            colors = colors.reshape(rays_o.shape[0], samples_per_ray, 3)
            
            # Execute volume rendering
            rgb_map, depth_map = self.execute_volume_rendering(
                densities, colors, z_vals, rays_d
            )
            
            # Reshape to image dimensions
            rgb_image = rgb_map.reshape(height, width, 3)
            depth_image = depth_map.reshape(height, width)
            
        return rgb_image, depth_image
