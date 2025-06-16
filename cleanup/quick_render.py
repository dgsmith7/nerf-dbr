#!/usr/bin/env python3
"""
Quick render script to preview NeRF quality during training.
This script loads the most recent checkpoint and renders a single novel view
to help assess current training quality without stopping the main training process.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.nerf import NeRFModel
from src.benchmark.pytorch_renderers import PyTorchMPSRenderer
from src.data.loader import load_synthetic_data
from src.training.trainer import NeRFTrainer


def find_latest_checkpoint(target_epoch=None):
    """Find the most recent checkpoint file or a specific epoch."""
    checkpoint_dir = Path("checkpoints")
    
    if target_epoch:
        # Look for specific epoch
        target_file = checkpoint_dir / f"checkpoint_epoch_{target_epoch}.pth"
        if target_file.exists():
            print(f"Found target checkpoint: {target_file} (Epoch {target_epoch})")
            return str(target_file), target_epoch
        else:
            raise FileNotFoundError(f"Checkpoint for epoch {target_epoch} not found")
    
    # Look for numbered checkpoints first
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
            print(f"Found latest checkpoint: {latest_path} (Epoch {latest_epoch})")
            return str(latest_path), latest_epoch
    
    # Fallback to final_model.pth
    final_model = checkpoint_dir / "final_model.pth"
    if final_model.exists():
        print(f"Using final model: {final_model}")
        return str(final_model), "final"
    
    raise FileNotFoundError("No checkpoint files found in checkpoints/")


def load_model_from_checkpoint(checkpoint_path):
    """Load the NeRF model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    coarse_model = NeRFModel().to(device)
    fine_model = NeRFModel().to(device)
    
    # Load weights - handle different checkpoint formats
    if 'coarse_model_state_dict' in checkpoint:
        # New format
        coarse_model.load_state_dict(checkpoint['coarse_model_state_dict'])
        fine_model.load_state_dict(checkpoint['fine_model_state_dict'])
    elif 'coarse_model' in checkpoint:
        # Current format
        coarse_model.load_state_dict(checkpoint['coarse_model'])
        fine_model.load_state_dict(checkpoint['fine_model'])
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
    
    print(f"Model loaded successfully on device: {device}")
    return coarse_model, fine_model, device


def render_test_view(checkpoint_path, device, resolution=(200, 200)):
    """Render a single test view for quality assessment."""
    
    # Load test data
    print("Loading test dataset...")
    datasets = load_synthetic_data('data/nerf_synthetic/lego')
    test_dataset = datasets['test']
    
    # Get a test example (middle of test set for representative view)
    test_idx = len(test_dataset) // 2
    test_data = test_dataset[test_idx]
    
    test_image = test_data['image']
    test_pose = test_data['pose'] 
    test_focal = test_data['focal']
    
    print(f"Rendering test view {test_idx} at resolution {resolution}...")
    
    # Create renderer and set it up properly
    renderer = PyTorchMPSRenderer()
    renderer.setup(checkpoint_path)  # This will load the models properly
    
    # Render the view
    with torch.no_grad():
        rgb_img, depth_img = renderer.render_image(test_pose, resolution)
    
    # Convert to numpy for display
    if isinstance(rgb_img, torch.Tensor):
        rgb_img = rgb_img.detach().cpu().numpy()
    if isinstance(depth_img, torch.Tensor):
        depth_img = depth_img.detach().cpu().numpy()
    if isinstance(test_image, torch.Tensor):
        test_image = test_image.detach().cpu().numpy()
    
    # Ensure proper range
    rgb_img = np.clip(rgb_img, 0, 1)
    test_image = np.clip(test_image, 0, 1)
    
    return rgb_img, depth_img, test_image


def calculate_quality_metrics(rendered_img, ground_truth_img):
    """Calculate quality metrics comparing rendered vs ground truth."""
    
    # Resize ground truth to match rendered resolution if needed
    if rendered_img.shape != ground_truth_img.shape:
        # Simple resize using torch
        import torch.nn.functional as F
        gt_tensor = torch.from_numpy(ground_truth_img).permute(2, 0, 1).unsqueeze(0)
        target_shape = rendered_img.shape[:2]
        gt_resized = F.interpolate(gt_tensor, size=target_shape, mode='bilinear', align_corners=False)
        ground_truth_resized = gt_resized.squeeze(0).permute(1, 2, 0).numpy()
    else:
        ground_truth_resized = ground_truth_img
    
    # Calculate MSE
    mse = np.mean((rendered_img - ground_truth_resized) ** 2)
    
    # Calculate PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # Calculate SSIM (simplified)
    mean_rendered = np.mean(rendered_img)
    mean_gt = np.mean(ground_truth_resized)
    var_rendered = np.var(rendered_img)
    var_gt = np.var(ground_truth_resized)
    cov = np.mean((rendered_img - mean_rendered) * (ground_truth_resized - mean_gt))
    
    ssim = (2 * mean_rendered * mean_gt + 0.01) * (2 * cov + 0.03) / \
           ((mean_rendered**2 + mean_gt**2 + 0.01) * (var_rendered + var_gt + 0.03))
    
    return mse, psnr, ssim


def save_comparison_image(rendered_img, depth_img, ground_truth_img, epoch_info, metrics):
    """Save a comparison image showing rendered vs ground truth."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(ground_truth_img)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # Rendered image
    axes[1].imshow(rendered_img)
    axes[1].set_title(f'NeRF Render (Epoch {epoch_info})', fontsize=14)
    axes[1].axis('off')
    
    # Depth map
    axes[2].imshow(depth_img, cmap='viridis')
    axes[2].set_title('Depth Map', fontsize=14)
    axes[2].axis('off')
    
    # Add metrics text - split between top and bottom to avoid overlap
    mse, psnr, ssim = metrics
    
    # Top title with PSNR (most important metric)
    fig.suptitle(f'NeRF Quality Assessment - Epoch {epoch_info} | PSNR: {psnr:.2f} dB', 
                fontsize=16, y=0.98)
    
    # Bottom subtitle with additional metrics
    fig.text(0.5, 0.02, f'MSE: {mse:.6f} | SSIM: {ssim:.3f}', 
             ha='center', va='bottom', fontsize=14, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Adjust layout to accommodate both titles
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save with timestamp
    output_file = f'quick_render_epoch_{epoch_info}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_file}")
    
    return output_file


def main():
    """Main function to render and evaluate current NeRF quality."""
    
    print("🎯 NeRF Quick Quality Assessment")
    print("=" * 50)
    
    try:
        # Check for command line argument for specific epoch
        import sys
        target_epoch = None
        if len(sys.argv) > 1:
            try:
                target_epoch = int(sys.argv[1])
                print(f"Targeting specific epoch: {target_epoch}")
            except ValueError:
                print("Invalid epoch number provided, using latest checkpoint")
        
        # Find checkpoint
        checkpoint_path, epoch_info = find_latest_checkpoint(target_epoch)
        
        # Render test view using the checkpoint
        rendered_img, depth_img, ground_truth_img = render_test_view(
            checkpoint_path, 'mps' if torch.backends.mps.is_available() else 'cpu', 
            resolution=(200, 200)
        )
        
        # Calculate metrics
        metrics = calculate_quality_metrics(rendered_img, ground_truth_img)
        mse, psnr, ssim = metrics
        
        # Print results
        print("\n🎉 QUALITY ASSESSMENT RESULTS")
        print("=" * 50)
        print(f"Checkpoint: Epoch {epoch_info}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"MSE:  {mse:.6f}")
        print(f"SSIM: {ssim:.3f}")
        print()
        
        # Interpret quality
        if psnr > 25:
            print("✅ EXCELLENT quality - renders are very sharp and accurate")
        elif psnr > 20:
            print("✅ GOOD quality - renders are clear with minor artifacts")
        elif psnr > 15:
            print("⚠️  FAIR quality - noticeable artifacts but recognizable")
        else:
            print("❌ POOR quality - significant artifacts, needs more training")
        
        # Training recommendation
        if mse < 0.005:
            print("🛑 RECOMMENDATION: Quality is excellent - safe to stop training")
        elif mse < 0.01:
            print("🤔 RECOMMENDATION: Good quality - can stop or continue for refinement")
        else:
            print("🚀 RECOMMENDATION: Continue training for better quality")
        
        # Save comparison image
        output_file = save_comparison_image(rendered_img, depth_img, ground_truth_img, 
                                          epoch_info, metrics)
        
        print(f"\n📸 Visual comparison saved to: {output_file}")
        print("🎯 Assessment complete!")
        
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
