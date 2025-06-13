#!/usr/bin/env python3
"""
Minimal integration test using actual lego dataset.
Tests the complete pipeline with minimal epochs and small resolution.
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))


def test_minimal_pipeline():
    """Test complete pipeline with minimal settings."""
    print("🧪 Running Minimal Integration Test")
    print("=" * 50)
    print("This test runs the complete pipeline with minimal settings:")
    print("• 2 training epochs")
    print("• 100x75 resolution")
    print("• 16 samples per ray")
    print("• Single camera view")
    print()
    
    # Check if lego data exists
    lego_path = Path("data/nerf_synthetic/lego")
    if not lego_path.exists():
        print("❌ Lego dataset not found at data/nerf_synthetic/lego")
        print("   Please download the synthetic dataset first.")
        return False
    
    try:
        # Import after path setup
        from src.data.loader import load_synthetic_data
        from src.training.trainer import NeRFTrainer
        from src.benchmark.benchmark_suite import UnifiedBenchmarkSuite
        
        print("✅ All imports successful")
        
        # Test device detection
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"✅ Using device: {device}")
        
        # Test dataset loading
        print("\n📊 Testing dataset loading...")
        datasets = load_synthetic_data(str(lego_path), device)
        
        if 'train' not in datasets:
            print("❌ Failed to load training dataset")
            return False
        
        train_dataset = datasets['train']
        print(f"✅ Loaded {len(train_dataset)} training images")
        
        # Test trainer setup
        print("\n🎯 Testing trainer setup...")
        config = {
            'device': device,
            'lr': 5e-4,
            'n_coarse': 16,      # Reduced for speed
            'n_fine': 32,        # Reduced for speed
            'chunk_size': 256,   # Reduced for speed
            'n_rays': 256,       # Reduced for speed
            'near': 2.0,
            'far': 6.0
        }
        
        trainer = NeRFTrainer(config)
        print("✅ Trainer initialized")
        
        # Test minimal training
        print("\n🏋️ Testing minimal training (2 epochs)...")
        start_time = time.time()
        
        trainer.train(train_dataset, val_dataset=None, n_epochs=2)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f}s")
        
        # Save test checkpoint
        test_checkpoint = "tests/outputs/minimal_test_model.pth"
        os.makedirs("tests/outputs", exist_ok=True)
        trainer.save_checkpoint("minimal_test_model.pth")
        
        # Move checkpoint to test directory
        import shutil
        shutil.move("checkpoints/minimal_test_model.pth", test_checkpoint)
        print(f"✅ Test model saved to {test_checkpoint}")
        
        # Test benchmark system
        print("\n⚡ Testing benchmark system...")
        suite = UnifiedBenchmarkSuite()
        suite.add_available_renderers()
        
        print(f"✅ Found {len(suite.renderers)} available renderers")
        
        # Quick benchmark test
        if len(suite.renderers) > 0:
            suite.run_benchmark(
                checkpoint_path=test_checkpoint,
                resolutions=[(100, 75)],  # Very small for speed
                samples_per_ray_options=[16],  # Minimal samples
                n_views=1  # Single view
            )
            
            results_df = suite.generate_report()
            print(f"✅ Benchmark completed with {len(results_df)} results")
        
        print("\n🎉 Minimal integration test PASSED!")
        print("System is working correctly end-to-end.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Minimal integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test outputs
        cleanup_test_files()


def cleanup_test_files():
    """Clean up test files to keep workspace tidy."""
    try:
        test_files = [
            "tests/outputs/minimal_test_model.pth",
            "unified_benchmark_results.csv",
            "unified_performance_comparison.png",
            "training_losses.png"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
        
        # Clean up sample renders if they exist
        sample_dir = Path("sample_renders")
        if sample_dir.exists():
            import shutil
            shutil.rmtree(sample_dir)
        
        print("🧹 Test files cleaned up")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run minimal integration test."""
    print("🚀 NeRF System - Minimal Integration Test")
    print()
    print("This test verifies the complete system works with real data.")
    print("It runs quickly with minimal settings and cleans up after itself.")
    print()
    
    # Create test output directory
    Path("tests/outputs").mkdir(parents=True, exist_ok=True)
    
    success = test_minimal_pipeline()
    
    if success:
        print("\n✅ SUCCESS: System is ready for full use!")
        print("\nNext steps:")
        print("  python main.py --epochs 20    # Full training")
        print("  python main.py --help         # See all options")
    else:
        print("\n❌ FAILED: Please check errors above and fix issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
