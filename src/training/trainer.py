"""
NeRF training implementation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path

from ..models.nerf import NeRFModel
from ..data.loader import SyntheticDataset
from ..utils.rendering import VolumeRenderer


class NeRFTrainer:
    """NeRF training class."""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize models with corrected architecture
        hidden_dim = config.get('hidden_dim', 256)  # Return to proven baseline
        pos_L = config.get('position_encoding_levels', 10)  # Return to standard
        dir_L = config.get('direction_encoding_levels', 4)  # Keep current
        
        self.coarse_model = NeRFModel(
            pos_L=pos_L, 
            dir_L=dir_L, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.fine_model = NeRFModel(
            pos_L=pos_L, 
            dir_L=dir_L, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Initialize renderer
        self.renderer = VolumeRenderer(self.device)
        
        # Initialize optimizers with weight decay
        params = list(self.coarse_model.parameters()) + list(self.fine_model.parameters())
        self.optimizer = optim.Adam(
            params, 
            lr=config.get('lr', 5e-4),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.get('lr_decay', 0.1)**(1/config.get('decay_steps', 250000))
        )
        
        # Training parameters
        self.n_coarse = config.get('n_coarse', 64)
        self.n_fine = config.get('n_fine', 128)
        self.chunk_size = config.get('chunk_size', 1024)
        self.near = config.get('near', 2.0)
        self.far = config.get('far', 6.0)
        
        # OPTIMIZED: Stability parameters
        self.gradient_clipping = config.get('gradient_clipping', None)
        self.checkpoint_frequency = config.get('checkpoint_frequency', 50)
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
        print(f"NeRF Trainer initialized on device: {self.device}")
    
    def train_step(self, batch: Dict) -> float:
        """
        Perform one training step.
        
        Args:
            batch: Training batch containing image, pose, and focal length
            
        Returns:
            Training loss for this batch
        """
        self.coarse_model.train()
        self.fine_model.train()
        
        # Move batch data to device
        image = batch['image'].to(self.device)  # [H, W, 3]
        pose = batch['pose'].to(self.device)    # [4, 4]
        focal = batch['focal']  # scalar, no need to move
        
        # Generate rays
        rays_o, rays_d = self._get_rays(pose, image.shape[:2], focal)
        
        # Flatten and select random rays
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target = image.reshape(-1, 3)
        
        # Random ray selection for training
        n_rays = self.config.get('n_rays', 1024)
        select_inds = torch.randperm(rays_o.shape[0], device=self.device)[:n_rays]
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        target = target[select_inds]
        
        # Render rays
        rgb_coarse, rgb_fine = self._render_rays(rays_o, rays_d)
        
        # Compute loss
        loss_coarse = F.mse_loss(rgb_coarse, target)
        loss_fine = F.mse_loss(rgb_fine, target)
        loss = loss_coarse + loss_fine
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # CORRECTED: Apply gradient clipping for stability
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
                self.gradient_clipping
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self, val_dataset: SyntheticDataset) -> float:
        """
        Validate model on validation set.
        
        Args:
            val_dataset: Validation dataset
            
        Returns:
            Average validation loss
        """
        self.coarse_model.eval()
        self.fine_model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for i in range(min(5, len(val_dataset))):  # Validate on subset
                batch = val_dataset[i]
                
                image = batch['image']
                pose = batch['pose']
                focal = batch['focal']
                
                # Render full image
                rgb_pred = self._render_image(pose, image.shape[:2], focal)
                
                # Compute loss
                loss = F.mse_loss(rgb_pred, image)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def train(self, train_dataset: SyntheticDataset, val_dataset: Optional[SyntheticDataset] = None,
              n_epochs: int = 100):
        """
        Train the NeRF model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            n_epochs: Number of training epochs
        """
        # Check for existing checkpoints to resume from
        start_epoch = 0
        latest_checkpoint = self._find_latest_checkpoint()
        
        if latest_checkpoint:
            print(f"Found checkpoint: {latest_checkpoint}")
            checkpoint_data = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            # Load model states
            self.coarse_model.load_state_dict(checkpoint_data['coarse_model'])
            self.fine_model.load_state_dict(checkpoint_data['fine_model'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer'])
            self.scheduler.load_state_dict(checkpoint_data['scheduler'])
            
            # Load training history
            self.train_losses = checkpoint_data.get('train_losses', [])
            self.val_losses = checkpoint_data.get('val_losses', [])
            
            # Determine start epoch
            start_epoch = len(self.train_losses)
            
            print(f"Resuming training from epoch {start_epoch + 1}/{n_epochs}")
            print(f"Previous training progress: {start_epoch}/{n_epochs} epochs ({100*start_epoch/n_epochs:.1f}%)")
        else:
            print(f"No checkpoint found. Starting training from epoch 1/{n_epochs}")
        
        # Skip training if already completed
        if start_epoch >= n_epochs:
            print(f"Training already completed! ({start_epoch}/{n_epochs} epochs)")
            return
        
        print(f"Training epochs {start_epoch + 1} to {n_epochs}...")
        
        for epoch in range(start_epoch, n_epochs):
            epoch_losses = []
            
            # Training loop
            progress_bar = tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{n_epochs}")
            
            for i in progress_bar:
                batch = train_dataset[i]
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
            
            # Record training loss
            avg_train_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_dataset is not None and (epoch + 1) % 10 == 0:
                val_loss = self.validate(val_dataset)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        print("Training completed!")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return None
        
        # Look for numbered checkpoints
        epoch_checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if epoch_checkpoints:
            # Sort by epoch number
            epoch_nums = []
            for cp in epoch_checkpoints:
                try:
                    epoch_num = int(cp.stem.split('_')[-1])
                    epoch_nums.append((epoch_num, cp))
                except ValueError:
                    continue
            
            if epoch_nums:
                epoch_nums.sort(key=lambda x: x[0])
                latest_epoch, latest_path = epoch_nums[-1]
                return str(latest_path)
        
        return None
    
    def _get_rays(self, pose: torch.Tensor, img_shape: tuple, focal: float):
        """Generate rays for given pose and image shape."""
        H, W = img_shape
        
        i, j = torch.meshgrid(
            torch.linspace(0, W-1, W, device=self.device),
            torch.linspace(0, H-1, H, device=self.device),
            indexing='ij'
        )
        i = i.t()
        j = j.t()
        
        dirs = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)
        
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
        rays_o = pose[:3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def _render_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        """Render rays using hierarchical sampling."""
        # Coarse sampling
        points_coarse, z_vals_coarse = self.renderer.sample_points_on_rays(
            rays_o, rays_d, self.near, self.far, self.n_coarse
        )
        
        # Query coarse network
        rgb_coarse = self._query_network(
            self.coarse_model, points_coarse, rays_d, z_vals_coarse
        )
        
        # Fine sampling (simplified - would use importance sampling in full implementation)
        points_fine, z_vals_fine = self.renderer.sample_points_on_rays(
            rays_o, rays_d, self.near, self.far, self.n_fine, perturb=False
        )
        
        # Query fine network
        rgb_fine = self._query_network(
            self.fine_model, points_fine, rays_d, z_vals_fine
        )
        
        return rgb_coarse, rgb_fine
    
    def _query_network(self, model: NeRFModel, points: torch.Tensor, 
                      ray_dirs: torch.Tensor, z_vals: torch.Tensor) -> torch.Tensor:
        """Query network and perform volume rendering."""
        # Flatten points for network query
        points_flat = points.reshape(-1, 3)
        dirs_flat = ray_dirs[:, None, :].expand_as(points).reshape(-1, 3)
        
        # Query network in chunks
        densities = []
        colors = []
        
        for i in range(0, points_flat.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, points_flat.shape[0])
            
            density_chunk, color_chunk = model(
                points_flat[i:end_idx], 
                dirs_flat[i:end_idx]
            )
            
            densities.append(density_chunk)
            colors.append(color_chunk)
        
        # Combine chunks
        densities = torch.cat(densities, dim=0)
        colors = torch.cat(colors, dim=0)
        
        # Reshape back
        densities = densities.reshape(*points.shape[:-1], 1)
        colors = colors.reshape(*points.shape)
        
        # Volume render
        rgb_map, _, _, _ = self.renderer.volume_render(densities, colors, z_vals, ray_dirs)
        
        return rgb_map
    
    def _render_image(self, pose: torch.Tensor, img_shape: tuple, focal: float) -> torch.Tensor:
        """Render full image."""
        H, W = img_shape
        rays_o, rays_d = self._get_rays(pose, img_shape, focal)
        
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Render in chunks
        rgb_chunks = []
        
        for i in range(0, rays_o.shape[0], self.chunk_size):
            end_idx = min(i + self.chunk_size, rays_o.shape[0])
            
            _, rgb_chunk = self._render_rays(rays_o[i:end_idx], rays_d[i:end_idx])
            rgb_chunks.append(rgb_chunk)
        
        rgb_img = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
        
        return rgb_img
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'coarse_model': self.coarse_model.state_dict(),
            'fine_model': self.fine_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, os.path.join('checkpoints', filename))
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.coarse_model.load_state_dict(checkpoint['coarse_model'])
        self.fine_model.load_state_dict(checkpoint['fine_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Checkpoint loaded: {filename}")
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        if self.val_losses:
            plt.subplot(1, 2, 2)
            plt.plot(self.val_losses)
            plt.title('Validation Loss')
            plt.xlabel('Epoch (x10)')
            plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_losses.png')
        plt.show()
