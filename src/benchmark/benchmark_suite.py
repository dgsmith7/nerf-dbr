"""
Unified benchmark suite for comparing NeRF execution methods.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .pytorch_renderers import PyTorchMPSRenderer, PyTorchCPURenderer, PyTorchCUDARenderer
from .numpy_renderer import NumPyRenderer


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method_name: str
    device: str
    resolution: str
    samples_per_ray: int
    render_time: float
    memory_mb: float
    rays_per_second: float
    device_info: str


class UnifiedBenchmarkSuite:
    """Benchmark suite for unified NeRF execution methods."""
    
    def __init__(self):
        self.renderers = []
        self.results = []
    
    def add_available_renderers(self):
        """Add all available renderers based on system capabilities."""
        print("Detecting available execution methods...")
        
        # Always available
        self.renderers.append(PyTorchCPURenderer())
        print("✓ PyTorch CPU renderer added")
        
        try:
            self.renderers.append(NumPyRenderer())
            print("✓ NumPy + Numba renderer added")
        except ImportError:
            print("✗ NumPy + Numba renderer failed (missing numba)")
        
        # MPS (Apple Silicon)
        try:
            self.renderers.append(PyTorchMPSRenderer())
            print("✓ PyTorch MPS renderer added")
        except RuntimeError:
            print("✗ MPS not available")
        
        # CUDA
        try:
            self.renderers.append(PyTorchCUDARenderer())
            print("✓ PyTorch CUDA renderer added")
        except RuntimeError:
            print("✗ CUDA not available")
        
        print(f"Total renderers: {len(self.renderers)}")
    
    def setup_renderers(self, checkpoint_path: str):
        """Setup all renderers with the shared trained model."""
        print(f"Setting up renderers with checkpoint: {checkpoint_path}")
        for renderer in self.renderers:
            renderer.setup(checkpoint_path)
    
    def generate_test_poses(self, n_views: int = 3) -> List[torch.Tensor]:
        """Generate test camera poses for benchmark."""
        poses = []
        
        for i in range(n_views):
            angle = i * 2 * np.pi / n_views
            
            # Create rotation around Y axis
            c2w = torch.eye(4, dtype=torch.float32)
            c2w[0, 0] = np.cos(angle)
            c2w[0, 2] = np.sin(angle)
            c2w[2, 0] = -np.sin(angle)
            c2w[2, 2] = np.cos(angle)
            c2w[2, 3] = 4.0  # Distance from origin
            
            poses.append(c2w)
        
        return poses
    
    def run_benchmark(self, 
                     checkpoint_path: str,
                     resolutions: List[Tuple[int, int]] = [(400, 300), (800, 600)],
                     samples_per_ray_options: List[int] = [64, 128],
                     n_views: int = 2):
        """
        Run unified benchmark across all execution methods.
        
        Args:
            checkpoint_path: Path to trained NeRF checkpoint
            resolutions: List of (width, height) tuples to test
            samples_per_ray_options: Different sample counts to test
            n_views: Number of camera views to test per configuration
        """
        # Setup renderers
        self.setup_renderers(checkpoint_path)
        
        # Generate test poses
        camera_poses = self.generate_test_poses(n_views)
        
        print("\nRunning unified NeRF benchmark...")
        print("=" * 60)
        print("This benchmark compares execution methods using the SAME trained model.")
        print("All methods use identical network weights and algorithms.")
        print("Only the execution environment differs.\n")
        
        total_configs = len(self.renderers) * len(resolutions) * len(samples_per_ray_options)
        config_count = 0
        
        for renderer in self.renderers:
            print(f"\nTesting {renderer.name}...")
            
            for resolution in resolutions:
                for samples_per_ray in samples_per_ray_options:
                    config_count += 1
                    print(f"  [{config_count}/{total_configs}] Resolution: {resolution}, Samples: {samples_per_ray}")
                    
                    # Run multiple views and average
                    times = []
                    memory_usages = []
                    
                    for view_idx, pose in enumerate(camera_poses):
                        try:
                            with renderer.performance_monitor():
                                rgb_img, depth_img = renderer.render_image(
                                    pose, resolution, samples_per_ray
                                )
                            
                            times.append(renderer.last_render_time)
                            memory_usages.append(renderer.peak_memory_mb)
                            
                            print(f"    View {view_idx + 1}: {renderer.last_render_time:.2f}s")
                            
                        except Exception as e:
                            print(f"    View {view_idx + 1}: FAILED - {e}")
                            continue
                    
                    if times:  # If at least one view succeeded
                        avg_time = np.mean(times)
                        avg_memory = np.mean(memory_usages)
                        total_rays = resolution[0] * resolution[1]
                        rays_per_second = total_rays / avg_time
                        
                        result = BenchmarkResult(
                            method_name=renderer.name,
                            device=renderer.device,
                            resolution=f"{resolution[0]}x{resolution[1]}",
                            samples_per_ray=samples_per_ray,
                            render_time=avg_time,
                            memory_mb=avg_memory,
                            rays_per_second=rays_per_second,
                            device_info=renderer.get_device_info()
                        )
                        
                        self.results.append(result)
                        
                        print(f"    Average: {avg_time:.2f}s, {rays_per_second:.0f} rays/s, {avg_memory:.1f}MB")
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive benchmark report."""
        if not self.results:
            print("No results to report")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'Method': r.method_name,
                'Device': r.device,
                'Resolution': r.resolution,
                'Samples/Ray': r.samples_per_ray,
                'Render Time (s)': r.render_time,
                'Memory (MB)': r.memory_mb,
                'Rays/Second': r.rays_per_second,
                'Device Info': r.device_info
            } for r in self.results
        ])
        
        # Create performance plots
        self._create_performance_plots(df)
        
        # Generate summary statistics
        summary = df.groupby(['Method', 'Device']).agg({
            'Render Time (s)': ['mean', 'std'],
            'Rays/Second': ['mean', 'std'],
            'Memory (MB)': ['mean', 'std']
        }).round(3)
        
        print("\n" + "="*80)
        print("UNIFIED NERF BENCHMARK RESULTS")
        print("="*80)
        print("\nDetailed Results:")
        print(df.to_string(index=False))
        
        print("\nSummary Statistics:")
        print(summary)
        
        # Performance insights
        if not df.empty:
            best_speed = df.loc[df['Rays/Second'].idxmax()]
            best_memory = df.loc[df['Memory (MB)'].idxmin()]
            
            print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
            print(f"   Fastest Method: {best_speed['Method']} ({best_speed['Rays/Second']:.0f} rays/sec)")
            print(f"   Most Memory Efficient: {best_memory['Method']} ({best_memory['Memory (MB)']:.1f} MB)")
            
            # Calculate speedup
            cpu_speed = df[df['Method'] == 'PyTorch CPU']['Rays/Second'].mean()
            if pd.notna(cpu_speed):
                max_speed = df['Rays/Second'].max()
                speedup = max_speed / cpu_speed
                print(f"   Maximum Speedup vs CPU: {speedup:.1f}x")
        
        # Save results
        df.to_csv('unified_benchmark_results.csv', index=False)
        print(f"\n📊 Results saved to:")
        print(f"   • unified_benchmark_results.csv")
        print(f"   • unified_performance_comparison.png")
        
        return df
    
    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Unified NeRF Execution Method Comparison', fontsize=16, fontweight='bold')
        
        # Color scheme for methods
        colors = {
            'PyTorch MPS': '#FF6B6B', 
            'PyTorch CPU': '#4ECDC4', 
            'PyTorch CUDA': '#45B7D1', 
            'NumPy + Numba': '#96CEB4'
        }
        
        # 1. Render time comparison
        pivot_time = df.pivot_table(values='Render Time (s)', 
                                   index='Resolution', 
                                   columns='Method', 
                                   aggfunc='mean')
        
        if not pivot_time.empty:
            pivot_time.plot(kind='bar', ax=axes[0,0], 
                           color=[colors.get(col, '#333333') for col in pivot_time.columns])
            axes[0,0].set_title('Average Render Time by Method')
            axes[0,0].set_ylabel('Time (seconds)')
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Throughput comparison
        pivot_rays = df.pivot_table(values='Rays/Second', 
                                   index='Resolution', 
                                   columns='Method', 
                                   aggfunc='mean')
        
        if not pivot_rays.empty:
            pivot_rays.plot(kind='bar', ax=axes[0,1],
                           color=[colors.get(col, '#333333') for col in pivot_rays.columns])
            axes[0,1].set_title('Throughput by Method')
            axes[0,1].set_ylabel('Rays/Second')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Memory usage comparison
        pivot_memory = df.pivot_table(values='Memory (MB)', 
                                     index='Resolution', 
                                     columns='Method', 
                                     aggfunc='mean')
        
        if not pivot_memory.empty:
            pivot_memory.plot(kind='bar', ax=axes[1,0],
                             color=[colors.get(col, '#333333') for col in pivot_memory.columns])
            axes[1,0].set_title('Memory Usage by Method')
            axes[1,0].set_ylabel('Memory (MB)')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Efficiency scatter plot
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            axes[1,1].scatter(method_data['Memory (MB)'], 
                            method_data['Rays/Second'], 
                            label=method, 
                            color=colors.get(method, '#333333'),
                            alpha=0.7, s=100)
        
        axes[1,1].set_xlabel('Memory Usage (MB)')
        axes[1,1].set_ylabel('Rays/Second')
        axes[1,1].set_title('Efficiency: Throughput vs Memory')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('unified_performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_sample_renders(self, checkpoint_path: str, output_dir: str = "sample_renders"):
        """Save sample renders from each method for visual comparison."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup renderers
        self.setup_renderers(checkpoint_path)
        
        # Generate a test pose
        test_pose = self.generate_test_poses(1)[0]
        resolution = (400, 300)
        samples_per_ray = 64
        
        print(f"\nGenerating sample renders in {output_dir}/...")
        
        for renderer in self.renderers:
            try:
                rgb_img, depth_img = renderer.render_image(test_pose, resolution, samples_per_ray)
                
                # Convert to numpy for saving
                rgb_np = rgb_img.cpu().numpy() if hasattr(rgb_img, 'cpu') else rgb_img
                depth_np = depth_img.cpu().numpy() if hasattr(depth_img, 'cpu') else depth_img
                
                # Save RGB
                plt.figure(figsize=(8, 6))
                plt.imshow(rgb_np)
                plt.title(f'{renderer.name} - RGB Render')
                plt.axis('off')
                plt.savefig(f'{output_dir}/{renderer.name.replace(" ", "_")}_rgb.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
                
                # Save depth
                plt.figure(figsize=(8, 6))
                plt.imshow(depth_np, cmap='viridis')
                plt.title(f'{renderer.name} - Depth Map')
                plt.colorbar()
                plt.axis('off')
                plt.savefig(f'{output_dir}/{renderer.name.replace(" ", "_")}_depth.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"  ✓ {renderer.name} renders saved")
                
            except Exception as e:
                print(f"  ✗ {renderer.name} failed: {e}")
        
        print("Sample renders completed!")
