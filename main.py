#!/usr/bin/env python3
"""
Main script for NeRF training and unified benchmark comparison.

This script:
1. Trains a NeRF model on synthetic data
2. Runs unified benchmark comparison across execution methods
3. Generates comprehensive performance reports
"""

import os
import sys
import torch
import argparse
from typing import Dict

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_synthetic_data
from src.training.trainer import NeRFTrainer
from src.benchmark.benchmark_suite import UnifiedBenchmarkSuite


def get_default_config() -> Dict:
    """Get optimized training configuration for convergence-first approach."""
    return {
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        
        # OPTIMIZED: Reduced learning rate for stability (was 5e-4)
        'lr': 1e-4,
        
        # OPTIMIZED: Gentler decay schedule for stability (was 0.1)
        'lr_decay': 0.1,
        
        # OPTIMIZED: Longer decay period for stability (was 250000)
        'decay_steps': 500000,
        
        # OPTIMIZED: Smaller batch size for stable gradients (was 1024)
        'n_rays': 1024,
        
        # Sampling configuration (maintain quality)
        'n_coarse': 64,
        'n_fine': 128,
        
        # OPTIMIZED: Enhanced model architecture
        'hidden_dim': 384,                   # Increased capacity (was 256)
        'position_encoding_levels': 8,       # Reduced for stability (was 10)
        'direction_encoding_levels': 4,      # Keep current
        
        # Memory configuration
        'chunk_size': 1024,
        
        # Scene bounds
        'near': 2.0,
        'far': 6.0,
        
        # NEW: Stability enhancements
        'gradient_clipping': 1.0,        # Prevent gradient explosion
        'weight_decay': 1e-6,            # Light regularization
        'checkpoint_frequency': 25,       # More frequent checkpoints
    }


def train_nerf(data_dir: str, config: Dict, n_epochs: int = 100) -> str:
    """
    Train NeRF model on synthetic dataset.
    
    Args:
        data_dir: Path to synthetic dataset directory
        config: Training configuration
        n_epochs: Number of training epochs
        
    Returns:
        Path to saved checkpoint
    """
    print("="*60)
    print("TRAINING NERF MODEL")
    print("="*60)
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    datasets = load_synthetic_data(data_dir, config['device'])
    
    if 'train' not in datasets:
        raise ValueError(f"Training dataset not found in {data_dir}")
    
    train_dataset = datasets['train']
    val_dataset = datasets.get('val', None)
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = NeRFTrainer(config)
    
    # Train model
    trainer.train(train_dataset, val_dataset, n_epochs)
    
    # Save final checkpoint
    checkpoint_path = f"checkpoints/final_model.pth"
    trainer.save_checkpoint("final_model.pth")
    
    # Plot training curves
    trainer.plot_losses()
    
    print(f"\nTraining completed! Model saved to: {checkpoint_path}")
    return checkpoint_path


def run_benchmark(checkpoint_path: str):
    """
    Run unified benchmark comparison.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
    """
    print("\n" + "="*60)
    print("UNIFIED BENCHMARK COMPARISON")
    print("="*60)
    
    # Initialize benchmark suite
    suite = UnifiedBenchmarkSuite()
    
    # Add available renderers
    suite.add_available_renderers()
    
    if not suite.renderers:
        print("No renderers available for benchmarking!")
        return
    
    # Configure benchmark
    resolutions = [
        (200, 150),   # Small - quick test
        (400, 300),   # Medium - standard test
        (800, 600),   # Large - performance test
    ]
    
    samples_per_ray_options = [32, 64, 128]
    n_views = 2
    
    print(f"\nBenchmark Configuration:")
    print(f"  Resolutions: {resolutions}")
    print(f"  Samples per ray: {samples_per_ray_options}")
    print(f"  Test views: {n_views}")
    print(f"  Total configurations: {len(suite.renderers) * len(resolutions) * len(samples_per_ray_options)}")
    
    # Run benchmark
    suite.run_benchmark(
        checkpoint_path=checkpoint_path,
        resolutions=resolutions,
        samples_per_ray_options=samples_per_ray_options,
        n_views=n_views
    )
    
    # Generate report
    results_df = suite.generate_report()
    
    # Print final summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)
    
    if not results_df.empty:
        print("\n🎯 KEY FINDINGS:")
        
        # Performance summary
        fastest_method = results_df.loc[results_df['Rays/Second'].idxmax(), 'Method']
        fastest_speed = results_df['Rays/Second'].max()
        
        cpu_results = results_df[results_df['Method'] == 'PyTorch CPU']
        if not cpu_results.empty:
            cpu_speed = cpu_results['Rays/Second'].mean()
            speedup = fastest_speed / cpu_speed
            print(f"   📈 Best Performance: {fastest_method} ({fastest_speed:.0f} rays/sec)")
            print(f"   🚀 Speedup vs CPU: {speedup:.1f}x")
        
        # Memory summary
        memory_efficient = results_df.loc[results_df['Memory (MB)'].idxmin(), 'Method']
        min_memory = results_df['Memory (MB)'].min()
        print(f"   💾 Most Memory Efficient: {memory_efficient} ({min_memory:.1f} MB)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   • For Development: Use {fastest_method} for fastest iteration")
        print(f"   • For Production: Balance performance vs hardware availability")
        print(f"   • For Mobile/Edge: Consider memory constraints ({memory_efficient})")
    
    print(f"\n📁 Output Files Generated:")
    print(f"   • checkpoints/final_model.pth - Trained model")
    print(f"   • unified_benchmark_results.csv - Detailed benchmark data")
    print(f"   • unified_performance_comparison.png - Performance charts")
    print(f"   • sample_renders/ - Visual quality comparison")
    print(f"   • training_losses.png - Training progress")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NeRF Training and Unified Benchmark")
    
    parser.add_argument('--data_dir', type=str, 
                       default='data/nerf_synthetic/lego',
                       help='Path to synthetic dataset directory')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and run benchmark only')
    
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/final_model.pth',
                       help='Path to model checkpoint (for benchmark only)')
    
    parser.add_argument('--benchmark_only', action='store_true',
                       help='Run benchmark only (alias for --skip_training)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('sample_renders', exist_ok=True)
    
    print("🚀 NeRF Training and Unified Benchmark System")
    print("="*60)
    print("This system will:")
    print("1. Train a NeRF model on synthetic data (if not skipped)")
    print("2. Compare execution methods using the SAME trained model")
    print("3. Generate comprehensive performance analysis")
    print("="*60)
    
    # Check if skipping training
    if args.skip_training or args.benchmark_only:
        print("Skipping training, running benchmark only...")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Please train a model first or provide a valid checkpoint path.")
            return
        
        checkpoint_path = args.checkpoint
    else:
        # Train model
        if not os.path.exists(args.data_dir):
            print(f"Error: Dataset directory not found at {args.data_dir}")
            print("Please ensure the synthetic dataset is available.")
            return
        
        config = get_default_config()
        print(f"Training configuration: {config}")
        
        checkpoint_path = train_nerf(args.data_dir, config, args.epochs)
    
    # Run benchmark
    run_benchmark(checkpoint_path)
    
    print("\n✅ All tasks completed successfully!")
    print("\nNext steps:")
    print("• Review the performance comparison charts")
    print("• Check sample renders for quality verification")
    print("• Use the fastest method for your specific use case")


if __name__ == "__main__":
    main()
