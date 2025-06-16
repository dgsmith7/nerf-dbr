#!/usr/bin/env python3
"""
Novel View Renderer Comparison Script
Renders a single novel view with each available renderer and displays them side-by-side.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.benchmark.pytorch_renderers import PyTorchMPSRenderer, PyTorchCPURenderer
    from src.benchmark.numpy_renderer import NumPyRenderer
    from src.benchmark.cpu_optimized_renderer import CPUOptimizedRenderer
    from src.benchmark.compressed_renderer import CompressedNeRFRenderer
    from src.data.loader import load_synthetic_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all renderer modules are available")


def find_best_checkpoint():
    """Find the best available checkpoint."""
    checkpoint_options = [
        "checkpoints/final_model.pth",
        "baseline_training_run_500epochs/checkpoints_baseline/final_model.pth",
        "baseline_training_run_500epochs/checkpoints_baseline/checkpoint_epoch_500.pth",
        "checkpoints/checkpoint_epoch_500.pth"
    ]
    
    for checkpoint_path in checkpoint_options:
        if os.path.exists(checkpoint_path):
            print(f"✅ Using checkpoint: {checkpoint_path}")
            return checkpoint_path
    
    raise FileNotFoundError("No trained model checkpoint found. Please train a model first.")


def create_novel_view_pose():
    """Create an interesting novel view pose for rendering."""
    # Create a simpler pose that's closer to training views
    pose = torch.eye(4, dtype=torch.float32)
    
    # Start with a basic pose - just move the camera back
    pose[0, 3] = 0.0   # x: centered
    pose[1, 3] = 0.0   # y: centered  
    pose[2, 3] = 3.0   # z: move back (closer than before)
    
    print(f"Using simpler novel view pose: camera at ({pose[0,3]:.1f}, {pose[1,3]:.1f}, {pose[2,3]:.1f})")
    
    return pose


def create_alternative_view_pose():
    """Create an alternative novel view pose."""
    # Try a different approach - slight offset from origin
    pose = torch.eye(4, dtype=torch.float32)
    
    # Very minimal movement
    pose[0, 3] = 0.5   # x: slight right
    pose[1, 3] = 0.0   # y: centered
    pose[2, 3] = 2.5   # z: closer
    
    return pose


def test_renderer(renderer_class, renderer_name, checkpoint_path, test_pose, resolution, samples, config=None):
    """Test a single renderer and return results."""
    print(f"Testing {renderer_name}...")
    
    try:
        # Initialize renderer
        if config:
            renderer = renderer_class(config)
        else:
            renderer = renderer_class()
        
        # Setup with checkpoint
        renderer.setup(checkpoint_path)
        
        # Render
        start_time = time.time()
        rgb, depth = renderer.render_image(test_pose, resolution, samples)
        render_time = time.time() - start_time
        
        # Convert to numpy if needed
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        
        # Ensure proper shape and range
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0, 1)
        
        # Debug: Check if image is completely black
        is_black = (rgb.max() < 0.01)
        mean_brightness = rgb.mean()
        
        print(f"   ✅ {renderer_name}: {render_time:.2f}s, RGB shape: {rgb.shape}")
        print(f"      Debug: max={rgb.max():.3f}, mean={mean_brightness:.3f}, black={is_black}")
        
        return {
            'success': True,
            'rgb': rgb,
            'depth': depth,
            'time': render_time,
            'name': renderer_name,
            'is_black': is_black,
            'mean_brightness': mean_brightness
        }
        
    except Exception as e:
        print(f"   ❌ {renderer_name} failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'time': float('inf'),
            'name': renderer_name
        }


def main():
    """Main comparison function."""
    print("🎬 Novel View Renderer Comparison")
    print("=" * 50)
    
    # Find checkpoint
    try:
        checkpoint_path = find_best_checkpoint()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Test configuration
    test_pose = create_novel_view_pose()
    resolution = (128, 128)  # Reasonable size for comparison
    samples = 32  # Enough for decent quality, fast enough for comparison
    
    print(f"Resolution: {resolution}")
    print(f"Samples per ray: {samples}")
    print()
    
    # Available renderers to test
    renderers_to_test = [
        (PyTorchMPSRenderer, "PyTorch MPS", None),
        (PyTorchCPURenderer, "PyTorch CPU", None),
    ]
    
    # Try to add other renderers if available
    try:
        renderers_to_test.append((NumPyRenderer, "NumPy + Numba", None))
    except NameError:
        print("⚠️  NumPy renderer not available")
    
    try:
        renderers_to_test.append((CPUOptimizedRenderer, "CPU Optimized", None))
    except NameError:
        print("⚠️  CPU Optimized renderer not available")
    
    try:
        compression_config = {
            'quantization_bits': 8,
            'pruning_ratio': 0.1,
            'use_mixed_precision': True,
            'compress_activations': True
        }
        renderers_to_test.append((CompressedNeRFRenderer, "Compressed NeRF", compression_config))
    except NameError:
        print("⚠️  Compressed NeRF renderer not available")
    
    # Test all renderers
    results = []
    for renderer_class, renderer_name, config in renderers_to_test:
        result = test_renderer(renderer_class, renderer_name, checkpoint_path, test_pose, resolution, samples, config)
        results.append(result)
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No renderers succeeded!")
        return
    
    print(f"\n✅ {len(successful_results)} renderers succeeded")
    
    # Create side-by-side comparison plot
    n_renderers = len(successful_results)
    fig, axes = plt.subplots(2, n_renderers, figsize=(4*n_renderers, 8))
    
    if n_renderers == 1:
        axes = axes.reshape(-1, 1)
    
    for i, result in enumerate(successful_results):
        # RGB image (top row)
        axes[0, i].imshow(result['rgb'])
        axes[0, i].set_title(f"{result['name']}\nRGB ({result['time']:.2f}s)")
        axes[0, i].axis('off')
        
        # Depth image (bottom row)
        depth_vis = result['depth']
        if depth_vis.max() > depth_vis.min():
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
        
        axes[1, i].imshow(depth_vis, cmap='viridis')
        axes[1, i].set_title(f"{result['name']}\nDepth")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    output_path = "outputs/renderer_comparison.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📸 Comparison saved to: {output_path}")
    
    # Close the plot to avoid interactive display issues
    plt.close()
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    successful_results.sort(key=lambda x: x['time'])
    
    for i, result in enumerate(successful_results):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{status} {result['name']:<20} {result['time']:6.2f}s")
    
    print(f"\n🎯 Comparison complete! Check {output_path} for saved results.")


if __name__ == "__main__":
    main()
