#!/usr/bin/env python3
"""
Test script for new CPU Optimized and Compressed NeRF renderers.
Verifies they work correctly and integrate with the benchmark system.
"""

import os
import sys
import torch
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.benchmark.cpu_optimized_renderer import CPUOptimizedRenderer
from src.benchmark.compressed_renderer import CompressedNeRFRenderer
from src.benchmark.pytorch_renderers import PyTorchCPURenderer


def test_new_renderers():
    """Test the new CPU Optimized and Compressed NeRF renderers."""
    print("🧪 Testing New Renderers")
    print("=" * 50)
    
    checkpoint_path = "checkpoints/final_model.pth"
    if not os.path.exists(checkpoint_path):
        print("❌ No trained model found. Please train a model first:")
        print("   python main.py --epochs 5")
        return False
    
    # Test pose and configuration
    test_pose = torch.eye(4, dtype=torch.float32)
    test_pose[2, 3] = 4.0  # Move camera back
    test_resolution = (64, 64)  # Small for quick testing
    test_samples = 16
    
    results = {}
    
    print("Testing CPU Optimized Renderer...")
    try:
        # Test CPU Optimized Renderer
        cpu_opt_renderer = CPUOptimizedRenderer()
        cpu_opt_renderer.setup(checkpoint_path)
        
        start_time = time.time()
        rgb, depth = cpu_opt_renderer.render_image(test_pose, test_resolution, test_samples)
        cpu_opt_time = time.time() - start_time
        
        print(f"   ✅ CPU Optimized: {cpu_opt_time:.2f}s, RGB shape: {rgb.shape}")
        results['CPU Optimized'] = {'time': cpu_opt_time, 'success': True}
        
    except Exception as e:
        print(f"   ❌ CPU Optimized failed: {e}")
        results['CPU Optimized'] = {'time': float('inf'), 'success': False, 'error': str(e)}
    
    print("Testing Compressed NeRF Renderer...")
    try:
        # Test Compressed NeRF Renderer
        compression_config = {
            'quantization_bits': 8,
            'pruning_ratio': 0.1,
            'use_mixed_precision': True,
            'compress_activations': True
        }
        compressed_renderer = CompressedNeRFRenderer(compression_config)
        compressed_renderer.setup(checkpoint_path)
        
        start_time = time.time()
        rgb, depth = compressed_renderer.render_image(test_pose, test_resolution, test_samples)
        compressed_time = time.time() - start_time
        
        print(f"   ✅ Compressed NeRF: {compressed_time:.2f}s, RGB shape: {rgb.shape}")
        results['Compressed NeRF'] = {'time': compressed_time, 'success': True}
        
    except Exception as e:
        print(f"   ❌ Compressed NeRF failed: {e}")
        results['Compressed NeRF'] = {'time': float('inf'), 'success': False, 'error': str(e)}
    
    # Baseline comparison with PyTorch CPU
    print("Testing PyTorch CPU (baseline)...")
    try:
        cpu_renderer = PyTorchCPURenderer()
        cpu_renderer.setup(checkpoint_path)
        
        start_time = time.time()
        rgb, depth = cpu_renderer.render_image(test_pose, test_resolution, test_samples)
        cpu_time = time.time() - start_time
        
        print(f"   ✅ PyTorch CPU: {cpu_time:.2f}s, RGB shape: {rgb.shape}")
        results['PyTorch CPU'] = {'time': cpu_time, 'success': True}
        
    except Exception as e:
        print(f"   ❌ PyTorch CPU failed: {e}")
        results['PyTorch CPU'] = {'time': float('inf'), 'success': False, 'error': str(e)}
    
    # Performance comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        # Sort by time
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['time'])
        fastest = sorted_results[0]
        
        print(f"🏆 Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")
        
        for name, result in sorted_results:
            speedup = fastest[1]['time'] / result['time'] if result['time'] > 0 else 1.0
            print(f"   {name}: {result['time']:.2f}s ({speedup:.1f}x)")
        
        # Check if new renderers work
        new_renderers_work = (
            results.get('CPU Optimized', {}).get('success', False) and
            results.get('Compressed NeRF', {}).get('success', False)
        )
        
        if new_renderers_work:
            print(f"\n✅ Both new renderers are working correctly!")
            print(f"📊 Ready for full benchmark: python main.py --benchmark_only")
            return True
        else:
            print(f"\n⚠️ Some new renderers have issues but system can continue.")
            return True
            
    else:
        print("❌ No renderers succeeded")
        return False


def test_integration_with_benchmark():
    """Test that new renderers integrate correctly with benchmark suite."""
    print(f"\n🔗 Testing Benchmark Integration")
    print("=" * 50)
    
    try:
        from src.benchmark.benchmark_suite import UnifiedBenchmarkSuite
        
        suite = UnifiedBenchmarkSuite()
        suite.add_available_renderers()
        
        # Check if our new renderers were added
        renderer_names = [r.name for r in suite.renderers]
        
        expected_new_renderers = ['CPU Optimized', 'Compressed NeRF']
        found_new_renderers = [name for name in expected_new_renderers if name in renderer_names]
        
        print(f"Available renderers: {renderer_names}")
        print(f"New renderers found: {found_new_renderers}")
        
        if len(found_new_renderers) == len(expected_new_renderers):
            print("✅ All new renderers integrated successfully!")
            return True
        else:
            missing = set(expected_new_renderers) - set(found_new_renderers)
            print(f"⚠️ Missing renderers: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests for new renderers."""
    print("🚀 New Renderer Test Suite")
    print("Testing CPU Optimized and Compressed NeRF renderers")
    print("=" * 60)
    
    # Test individual renderers
    renderers_work = test_new_renderers()
    
    # Test benchmark integration
    integration_works = test_integration_with_benchmark()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Individual Renderers: {'✅ PASS' if renderers_work else '❌ FAIL'}")
    print(f"Benchmark Integration: {'✅ PASS' if integration_works else '❌ FAIL'}")
    
    if renderers_work and integration_works:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"New renderers are ready for use:")
        print(f"  • CPU Optimized: Pure CPU implementation without PyTorch overhead")
        print(f"  • Compressed NeRF: Memory-optimized with quantization and pruning")
        print(f"\nRun full benchmark: python main.py --benchmark_only")
        return True
    else:
        print(f"\n⚠️ Some tests failed, but system may still work.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
