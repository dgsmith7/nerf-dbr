"""
Compressed NeRF renderer using quantized and pruned network weights.

This renderer demonstrates memory optimization techniques for NeRF deployment:
- Weight quantization (float32 → int8/int16)
- Weight pruning (removing small weights)
- Reduced precision computations

The same network architecture is used but with compressed weights to show
memory vs accuracy trade-offs for mobile/edge deployment scenarios.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import copy

from .base_renderer import BaseUnifiedRenderer


class CompressedNeRFRenderer(BaseUnifiedRenderer):
    """Memory-optimized NeRF using quantization and pruning."""
    
    def __init__(self, compression_config: Dict[str, Any] = None):
        super().__init__("Compressed NeRF", "cpu")
        
        # Default compression configuration
        self.config = compression_config or {
            'quantization_bits': 8,      # 8-bit quantization (int8)
            'pruning_ratio': 0.1,        # Remove 10% of smallest weights
            'use_mixed_precision': True,  # Use float16 for some operations
            'compress_activations': True  # Quantize intermediate activations
        }
        
        # Compressed model storage
        self.compressed_coarse = None
        self.compressed_fine = None
        
        # Quantization parameters
        self.quant_params = {}
        
    def setup(self, checkpoint_path: str):
        """Setup renderer with compressed model weights."""
        super().setup(checkpoint_path)
        
        # Get original PyTorch models
        coarse_model, fine_model = self.shared_model.get_models(self.device)
        
        print(f"Compressing models with config: {self.config}")
        
        # Compress both models
        self.compressed_coarse = self._compress_model(coarse_model, "coarse")
        self.compressed_fine = self._compress_model(fine_model, "fine")
        
        # Calculate compression statistics
        self._print_compression_stats(coarse_model, fine_model)
    
    def positional_encoding(self, x: torch.Tensor, L: int) -> torch.Tensor:
        """Apply positional encoding to input coordinates."""
        encoded = [x]
        for i in range(L):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * torch.pi * x))
            encoded.append(torch.cos(freq * torch.pi * x))
        return torch.cat(encoded, dim=-1)
        
    def _compress_model(self, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
        """Apply compression techniques to a PyTorch model."""
        compressed_model = {}
        
        for name, param in model.named_parameters():
            weight = param.detach().cpu()
            
            # Apply pruning
            if self.config['pruning_ratio'] > 0:
                weight = self._prune_weights(weight, self.config['pruning_ratio'])
            
            # Apply quantization
            if self.config['quantization_bits'] < 32:
                weight, quant_params = self._quantize_weights(
                    weight, self.config['quantization_bits']
                )
                self.quant_params[f"{model_name}_{name}"] = quant_params
            
            compressed_model[name] = weight
            
        return compressed_model
    
    def _prune_weights(self, weight: torch.Tensor, pruning_ratio: float) -> torch.Tensor:
        """Remove smallest weights by magnitude."""
        if pruning_ratio <= 0:
            return weight
            
        # Calculate threshold for pruning
        flat_weights = weight.flatten().abs()
        threshold = torch.quantile(flat_weights, pruning_ratio)
        
        # Create mask for weights to keep
        mask = weight.abs() > threshold
        
        # Apply pruning
        pruned_weight = weight * mask.float()
        
        return pruned_weight
    
    def _quantize_weights(self, weight: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Quantize weights to specified bit precision."""
        if bits == 32:
            return weight, {}
        
        # Calculate quantization parameters
        w_min = weight.min().item()
        w_max = weight.max().item()
        
        # Avoid division by zero
        if w_max == w_min:
            return weight, {'scale': 1.0, 'zero_point': 0.0, 'min': w_min, 'max': w_max}
        
        # Calculate scale and zero point
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        zero_point = torch.round(torch.clamp(torch.tensor(zero_point), qmin, qmax)).int()
        
        # Quantize
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Store as appropriate integer type
        if bits <= 8:
            quantized = quantized.to(torch.int8)
        else:
            quantized = quantized.to(torch.int16)
        
        quant_params = {
            'scale': scale,
            'zero_point': zero_point.item(),
            'min': w_min,
            'max': w_max,
            'bits': bits
        }
        
        return quantized, quant_params
    
    def _dequantize_weights(self, quantized_weight: torch.Tensor, 
                           quant_params: Dict[str, float]) -> torch.Tensor:
        """Convert quantized weights back to float for computation."""
        if not quant_params:
            return quantized_weight.float()
        
        scale = quant_params['scale']
        zero_point = quant_params['zero_point']
        
        # Dequantize
        dequantized = (quantized_weight.float() - zero_point) * scale
        
        return dequantized
    
    def _compressed_mlp_forward(self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor,
                               compressed_weights: Dict[str, torch.Tensor],
                               model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using compressed weights."""
        
        def compressed_linear(input_tensor, weight_name, bias_name):
            # Get compressed weights
            weight = compressed_weights[weight_name]
            bias = compressed_weights[bias_name]
            
            # Dequantize if needed
            param_key = f"{model_name}_{weight_name}"
            if param_key in self.quant_params:
                weight = self._dequantize_weights(weight, self.quant_params[param_key])
            
            param_key = f"{model_name}_{bias_name}"
            if param_key in self.quant_params:
                bias = self._dequantize_weights(bias, self.quant_params[param_key])
            
            # Use mixed precision if enabled
            if self.config['use_mixed_precision']:
                weight = weight.half()
                input_tensor = input_tensor.half()
                result = torch.nn.functional.linear(input_tensor, weight, bias.half())
                return result.float()
            else:
                return torch.nn.functional.linear(input_tensor, weight, bias)
        
        # Forward pass through compressed network (pos_encoded only for first layers)
        x1 = torch.relu(compressed_linear(pos_encoded, 'layers.0.weight', 'layers.0.bias'))
        x2 = torch.relu(compressed_linear(x1, 'layers.1.weight', 'layers.1.bias'))
        x3 = torch.relu(compressed_linear(x2, 'layers.2.weight', 'layers.2.bias'))
        x4 = torch.relu(compressed_linear(x3, 'layers.3.weight', 'layers.3.bias'))
        
        # Skip connection: concatenate x4 with pos_encoded
        x4_skip = torch.cat([x4, pos_encoded], dim=1)
        
        x5 = torch.relu(compressed_linear(x4_skip, 'layers.4.weight', 'layers.4.bias'))
        x6 = torch.relu(compressed_linear(x5, 'layers.5.weight', 'layers.5.bias'))
        x7 = torch.relu(compressed_linear(x6, 'layers.6.weight', 'layers.6.bias'))
        x8 = torch.relu(compressed_linear(x7, 'layers.7.weight', 'layers.7.bias'))
        
        # Output layers
        density = torch.relu(compressed_linear(x8, 'density_head.weight', 'density_head.bias'))
        
        # Color branch: concatenate x8 with dir_encoded
        color_input = torch.cat([x8, dir_encoded], dim=1)
        color_hidden = torch.relu(compressed_linear(color_input, 'color_layers.0.weight', 'color_layers.0.bias'))
        color = torch.sigmoid(compressed_linear(color_hidden, 'color_layers.1.weight', 'color_layers.1.bias'))
        
        return density, color
    
    def query_nerf_networks(self, positions: torch.Tensor, directions: torch.Tensor,
                           use_fine: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query compressed NeRF networks."""
        
        # Use positional encoding
        pos_encoded = self.positional_encoding(positions, L=10)  # 63 features
        dir_encoded = self.positional_encoding(directions, L=4)   # 27 features
        
        # Query compressed networks - use fine network by default
        if use_fine:
            density, color = self._compressed_mlp_forward(
                pos_encoded, dir_encoded, self.compressed_fine, "fine"
            )
        else:
            density, color = self._compressed_mlp_forward(
                pos_encoded, dir_encoded, self.compressed_coarse, "coarse"
            )
        
        return density, color
    
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Volume rendering with potential activation compression."""
        
        # Standard volume rendering computation (same as PyTorch MPS renderer)
        import torch.nn.functional as F
        
        # Distance between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        # Use a smaller value for half precision compatibility
        inf_val = 1e4 if self.config['compress_activations'] else 1e10
        dists = torch.cat([dists, torch.full_like(dists[..., :1], inf_val)], dim=-1)
        dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
        
        # Apply compression after distance computation if enabled
        if self.config['compress_activations']:
            densities = densities.half()
            colors = colors.half()
            dists = dists.half()
        
        # Alpha compositing
        alpha = 1.0 - torch.exp(-F.relu(densities[..., 0]) * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), 
                                 transmittance[..., :-1]], dim=-1)
        
        weights = alpha * transmittance
        
        # Composite colors and depth
        rgb_map = torch.sum(weights[..., None] * colors, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        if self.config['compress_activations']:
            # Convert back to float
            return rgb_map.float(), depth_map.float()
        else:
            return rgb_map, depth_map
    
    def _print_compression_stats(self, original_coarse: torch.nn.Module, 
                                original_fine: torch.nn.Module):
        """Print compression statistics."""
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def estimate_memory(compressed_weights, quant_params):
            total_memory = 0
            for name, weight in compressed_weights.items():
                if f"coarse_{name}" in quant_params or f"fine_{name}" in quant_params:
                    # Quantized weights use less memory
                    bits = self.config['quantization_bits']
                    total_memory += weight.numel() * bits / 8
                else:
                    # Full precision weights
                    total_memory += weight.numel() * 4  # float32
            return total_memory
        
        # Calculate original statistics
        original_params = count_parameters(original_coarse) + count_parameters(original_fine)
        original_memory = original_params * 4  # float32
        
        # Calculate compressed statistics
        compressed_memory = (
            estimate_memory(self.compressed_coarse, self.quant_params) +
            estimate_memory(self.compressed_fine, self.quant_params)
        )
        
        compression_ratio = original_memory / compressed_memory
        memory_reduction = (1 - compressed_memory / original_memory) * 100
        
        print(f"📊 Compression Statistics:")
        print(f"   Original memory: {original_memory / 1024 / 1024:.2f} MB")
        print(f"   Compressed memory: {compressed_memory / 1024 / 1024:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Memory reduction: {memory_reduction:.1f}%")
        print(f"   Quantization: {self.config['quantization_bits']}-bit")
        print(f"   Pruning ratio: {self.config['pruning_ratio'] * 100:.1f}%")
    
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render full image using compressed NeRF networks.
        
        Args:
            camera_pose: Camera-to-world transformation [4, 4]
            resolution: Image resolution (width, height)
            samples_per_ray: Number of samples per ray
            
        Returns:
            (rgb_image, depth_image): Rendered outputs
        """
        width, height = resolution
        
        with self.performance_monitor():
            # Generate rays
            rays_o, rays_d = self.generate_rays(camera_pose, width, height)
            
            # Flatten for processing
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            
            # Sample points along rays
            points, z_vals = self.sample_points_on_rays(rays_o_flat, rays_d_flat, samples_per_ray)
            
            # Reshape for network queries
            n_rays = rays_o_flat.shape[0]
            points_flat = points.reshape(-1, 3)
            dirs_flat = rays_d_flat.unsqueeze(1).expand(-1, samples_per_ray, -1).reshape(-1, 3)
            
            # Query networks for all points
            densities, colors = self.query_nerf_networks(points_flat, dirs_flat)
            
            # Reshape back
            densities = densities.reshape(n_rays, samples_per_ray, -1)
            colors = colors.reshape(n_rays, samples_per_ray, 3)
            
            # Volume rendering
            rgb_flat, depth_flat = self.execute_volume_rendering(
                densities, colors, z_vals, rays_d_flat
            )
            
            # Reshape to image
            rgb_image = rgb_flat.reshape(height, width, 3)
            depth_image = depth_flat.reshape(height, width)
        
        return rgb_image, depth_image
