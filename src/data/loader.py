"""
Data loading utilities for NeRF synthetic dataset.
"""

import json
import os
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional


class SyntheticDataset:
    """Dataset loader for NeRF synthetic data format."""
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 img_wh: Tuple[int, int] = (800, 800), device: str = 'cpu'):
        """
        Args:
            data_dir: Path to synthetic dataset directory
            split: 'train', 'val', or 'test'
            img_wh: Image dimensions (width, height)
            device: Device to load tensors on
        """
        self.data_dir = data_dir
        self.split = split
        self.img_w, self.img_h = img_wh
        self.device = device
        
        # Load transforms file
        transforms_file = os.path.join(data_dir, f'transforms_{split}.json')
        with open(transforms_file, 'r') as f:
            self.meta = json.load(f)
        
        # Extract camera parameters
        self.focal = 0.5 * self.img_w / np.tan(0.5 * self.meta['camera_angle_x'])
        
        # Load images and poses
        self.images = []
        self.poses = []
        
        for frame in self.meta['frames']:
            # Load image
            img_path = os.path.join(data_dir, frame['file_path'] + '.png')
            img = Image.open(img_path).convert('RGBA')
            img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
            
            # Convert to numpy and handle alpha channel
            img_array = np.array(img) / 255.0
            if img_array.shape[-1] == 4:  # RGBA
                # Composite on white background
                rgb = img_array[..., :3]
                alpha = img_array[..., 3:4]
                img_array = rgb * alpha + (1 - alpha)
            
            self.images.append(img_array)
            
            # Load pose
            pose = np.array(frame['transform_matrix'])
            self.poses.append(pose)
        
        # Convert to tensors
        self.images = torch.FloatTensor(np.stack(self.images)).to(device)
        self.poses = torch.FloatTensor(np.stack(self.poses)).to(device)
        
        print(f"Loaded {len(self.images)} images from {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'pose': self.poses[idx],
            'focal': self.focal
        }
    
    def get_rays(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for a given camera pose.
        
        Args:
            pose: Camera-to-world transformation matrix [4, 4]
            
        Returns:
            (rays_o, rays_d): Ray origins and directions
        """
        # Create coordinate grids
        i, j = torch.meshgrid(
            torch.linspace(0, self.img_w-1, self.img_w, device=self.device),
            torch.linspace(0, self.img_h-1, self.img_h, device=self.device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        # Convert to normalized device coordinates
        dirs = torch.stack([
            (i - self.img_w * 0.5) / self.focal,
            -(j - self.img_h * 0.5) / self.focal,
            -torch.ones_like(i)
        ], dim=-1)
        
        # Apply camera-to-world transformation
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d


def load_synthetic_data(data_dir: str, device: str = 'cpu') -> Dict[str, SyntheticDataset]:
    """
    Load all splits of synthetic dataset.
    
    Args:
        data_dir: Path to dataset directory
        device: Device to load data on
        
    Returns:
        Dictionary with train/val/test datasets
    """
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = SyntheticDataset(data_dir, split, device=device)
        except FileNotFoundError:
            print(f"Warning: {split} split not found in {data_dir}")
    
    return datasets
