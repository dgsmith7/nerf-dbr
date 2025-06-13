"""
Base class for unified NeRF renderers.

All renderers use the same trained NeRF model but execute with different methods.
"""

import torch
import time
import psutil
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from contextlib import contextmanager


class SharedNeRFModel:
    """Singleton to ensure all renderers use the same trained model."""
    
    _instance = None
    _models_by_device = {}  # Store models for each device
    _loaded_checkpoint = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_models(self, checkpoint_path: str, device: str = 'cpu'):
        """Load shared NeRF models from checkpoint for specific device."""
        device_key = device
        
        # Check if we need to load models for this device/checkpoint combination
        if (device_key not in self._models_by_device or 
            self._loaded_checkpoint != checkpoint_path):
            
            print(f"Loading shared NeRF models from {checkpoint_path} for device: {device}")
            
            try:
                from ..models.nerf import NeRFModel
                
                # Load checkpoint with weights_only=False for compatibility with NumPy objects
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                coarse_model = NeRFModel().to(device)
                fine_model = NeRFModel().to(device)
                
                coarse_model.load_state_dict(checkpoint['coarse_model'])
                fine_model.load_state_dict(checkpoint['fine_model'])
                
                coarse_model.eval()
                fine_model.eval()
                
                # Store models for this device
                self._models_by_device[device_key] = {
                    'coarse': coarse_model,
                    'fine': fine_model
                }
                
                self._loaded_checkpoint = checkpoint_path
                print("Shared models loaded successfully")
                
            except FileNotFoundError:
                print("Checkpoint not found, using randomly initialized models")
                from ..models.nerf import NeRFModel
                
                coarse_model = NeRFModel().to(device)
                fine_model = NeRFModel().to(device)
                
                coarse_model.eval()
                fine_model.eval()
                
                # Store models for this device
                self._models_by_device[device_key] = {
                    'coarse': coarse_model,
                    'fine': fine_model
                }
        else:
            print(f"Using cached models for device: {device}")
    
    def get_models(self, device: str = 'cpu'):
        """Get the shared model instances for specific device."""
        device_key = device
        if device_key not in self._models_by_device:
            raise RuntimeError(f"Models not loaded for device {device}. Call load_models() first.")
        
        models = self._models_by_device[device_key]
        return models['coarse'], models['fine']


class BaseUnifiedRenderer(ABC):
    """Base class for unified NeRF renderers."""
    
    def __init__(self, name: str, device: str = 'cpu'):
        """
        Args:
            name: Renderer name for identification
            device: Device to run on ('cpu', 'mps', 'cuda')
        """
        self.name = name
        self.device = device
        self.shared_model = SharedNeRFModel()
        
        # Performance monitoring
        self.last_render_time = 0.0
        self.peak_memory_mb = 0.0
        self._monitoring = False
        
        # Scene parameters
        self.near = 2.0
        self.far = 6.0
        
        print(f"Initialized {self.name} renderer on {device}")
    
    def setup(self, checkpoint_path: str):
        """Setup renderer with trained model."""
        self.shared_model.load_models(checkpoint_path, self.device)
    
    @contextmanager
    def performance_monitor(self):
        """Monitor execution time and memory usage."""
        # Start memory monitoring
        memory_thread = threading.Thread(target=self._monitor_memory)
        self.peak_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self._monitoring = True
        memory_thread.start()
        
        # Synchronize device
        if self.device == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif self.device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        yield
        
        # End timing
        if self.device == 'mps' and torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        self.last_render_time = end_time - start_time
        
        # Stop memory monitoring
        self._monitoring = False
        memory_thread.join()
    
    def _monitor_memory(self):
        """Monitor peak memory usage in background thread."""
        while self._monitoring:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            time.sleep(0.01)
    
    def get_device_info(self) -> str:
        """Get device information string."""
        if self.device == 'mps':
            return "Apple M3 Pro (MPS)"
        elif self.device == 'cuda':
            return f"CUDA - {torch.cuda.get_device_name()}"
        else:
            return f"CPU - {psutil.cpu_count()} cores"
    
    def query_nerf_networks(self, positions: torch.Tensor, directions: torch.Tensor,
                           use_fine: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the shared NeRF networks.
        
        Args:
            positions: 3D positions [N, 3]
            directions: Viewing directions [N, 3]
            use_fine: Whether to use fine network (True) or coarse (False)
            
        Returns:
            (density, rgb): Volume density and RGB color
        """
        coarse_model, fine_model = self.shared_model.get_models(self.device)
        model = fine_model if use_fine else coarse_model
        
        # Ensure tensors are on correct device
        positions = positions.to(self.device)
        directions = directions.to(self.device)
        
        with torch.no_grad():
            density, rgb = model(positions, directions)
        
        return density, rgb
    
    @abstractmethod
    def execute_volume_rendering(self, densities: torch.Tensor, colors: torch.Tensor,
                                z_vals: torch.Tensor, ray_directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute volume rendering - this is where execution methods differ.
        
        Args:
            densities: Volume densities [N, n_samples, 1]
            colors: RGB colors [N, n_samples, 3]
            z_vals: Depth values [N, n_samples]
            ray_directions: Ray directions [N, 3]
            
        Returns:
            (rgb_map, depth_map): Rendered RGB and depth
        """
        pass
    
    @abstractmethod
    def render_image(self, camera_pose: torch.Tensor, resolution: Tuple[int, int],
                    samples_per_ray: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render full image.
        
        Args:
            camera_pose: Camera-to-world transformation [4, 4]
            resolution: Image resolution (width, height)
            samples_per_ray: Number of samples per ray
            
        Returns:
            (rgb_image, depth_image): Rendered outputs
        """
        pass
    
    def generate_rays(self, camera_pose: torch.Tensor, width: int, height: int,
                     focal: float = 800.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate camera rays for given pose and image dimensions.
        
        Args:
            camera_pose: Camera-to-world transformation [4, 4]
            width: Image width
            height: Image height
            focal: Focal length
            
        Returns:
            (rays_o, rays_d): Ray origins and directions
        """
        # Create coordinate grids
        i, j = torch.meshgrid(
            torch.linspace(0, width-1, width, device=self.device),
            torch.linspace(0, height-1, height, device=self.device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        # Convert to normalized device coordinates
        dirs = torch.stack([
            (i - width * 0.5) / focal,
            -(j - height * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)
        
        # Apply camera-to-world transformation
        camera_pose = camera_pose.to(self.device)
        rays_d = torch.sum(dirs[..., None, :] * camera_pose[:3, :3], dim=-1)
        rays_o = camera_pose[:3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def sample_points_on_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                             n_samples: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3]
            n_samples: Number of samples per ray
            
        Returns:
            (points, z_vals): Sampled points and depth values
        """
        # Linear sampling in depth
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=self.device)
        z_vals = self.near * (1.0 - t_vals) + self.far * t_vals
        z_vals = z_vals.expand(rays_o.shape[0], n_samples)
        
        # Get 3D points along rays
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        
        return points, z_vals
