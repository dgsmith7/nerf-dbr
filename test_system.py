#!/usr/bin/env python3
"""
Focused testing suite for NeRF system.

Tests core functionality with minimal workspace impact.
All test outputs go to tests/outputs/ and are cleaned up automatically.
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.nerf import NeRFModel, PositionalEncoding
from src.utils.rendering import VolumeRenderer
from src.benchmark.base_renderer import SharedNeRFModel, BaseUnifiedRenderer


class TestResults:
    """Clean test results tracking."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def record(self, test_name: str, passed: bool, message: str = ""):
        """Record test result."""
        self.tests.append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
        
        if passed:
            self.passed += 1
            print(f"✅ {test_name}")
        else:
            self.failed += 1
            print(f"❌ {test_name}: {message}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n📊 Test Results: {self.passed}/{total} passed")
        
        if self.failed > 0:
            print("\n❌ Failed tests:")
            for test in self.tests:
                if not test['passed']:
                    print(f"   • {test['name']}: {test['message']}")
        
        return self.failed == 0


def test_positional_encoding():
    """Test positional encoding functionality."""
    results = TestResults()
    
    try:
        # Test basic encoding
        encoder = PositionalEncoding(L=3)
        input_coords = torch.randn(10, 3)
        encoded = encoder.encode(input_coords)
        
        # Check output shape
        expected_dim = 3 + 3 * 2 * 3  # 3 + 18 = 21
        if encoded.shape == (10, expected_dim):
            results.record("positional_encoding_shape", True)
        else:
            results.record("positional_encoding_shape", False, 
                         f"Expected shape (10, {expected_dim}), got {encoded.shape}")
        
        # Test numpy input
        import numpy as np
        np_input = np.random.randn(5, 3)
        encoded_np = encoder.encode(np_input)
        
        if encoded_np.shape == (5, expected_dim):
            results.record("positional_encoding_numpy", True)
        else:
            results.record("positional_encoding_numpy", False, "NumPy input failed")
        
    except Exception as e:
        results.record("positional_encoding_basic", False, str(e))
    
    return results


def test_nerf_model():
    """Test NeRF model functionality."""
    results = TestResults()
    
    try:
        # Test model creation
        model = NeRFModel()
        results.record("nerf_model_creation", True)
        
        # Test forward pass
        positions = torch.randn(100, 3)
        directions = torch.randn(100, 3)
        
        density, rgb = model(positions, directions)
        
        # Check output shapes
        if density.shape == (100, 1):
            results.record("nerf_density_shape", True)
        else:
            results.record("nerf_density_shape", False, f"Expected (100, 1), got {density.shape}")
        
        if rgb.shape == (100, 3):
            results.record("nerf_rgb_shape", True)
        else:
            results.record("nerf_rgb_shape", False, f"Expected (100, 3), got {rgb.shape}")
        
        # Check value ranges
        if torch.all(density >= 0):
            results.record("nerf_density_positive", True)
        else:
            results.record("nerf_density_positive", False, "Density contains negative values")
        
        if torch.all(rgb >= 0) and torch.all(rgb <= 1):
            results.record("nerf_rgb_range", True)
        else:
            results.record("nerf_rgb_range", False, "RGB values outside [0,1] range")
        
    except Exception as e:
        results.record("nerf_model_forward", False, str(e))
    
    return results


def test_volume_renderer():
    """Test volume rendering functionality."""
    results = TestResults()
    
    try:
        renderer = VolumeRenderer('cpu')
        
        # Test point sampling
        rays_o = torch.zeros(10, 3)
        rays_d = torch.tensor([[0, 0, -1]] * 10).float()
        
        points, z_vals = renderer.sample_points_on_rays(rays_o, rays_d, n_samples=32)
        
        if points.shape == (10, 32, 3):
            results.record("volume_sampling_shape", True)
        else:
            results.record("volume_sampling_shape", False, f"Expected (10, 32, 3), got {points.shape}")
        
        # Test volume rendering
        densities = torch.rand(10, 32, 1)
        colors = torch.rand(10, 32, 3)
        
        rgb_map, depth_map, acc_map, weights = renderer.volume_render(
            densities, colors, z_vals, rays_d
        )
        
        if rgb_map.shape == (10, 3):
            results.record("volume_render_rgb_shape", True)
        else:
            results.record("volume_render_rgb_shape", False, f"Expected (10, 3), got {rgb_map.shape}")
        
        if depth_map.shape == (10,):
            results.record("volume_render_depth_shape", True)
        else:
            results.record("volume_render_depth_shape", False, f"Expected (10,), got {depth_map.shape}")
        
    except Exception as e:
        results.record("volume_renderer_basic", False, str(e))
    
    return results


def test_shared_model():
    """Test shared model singleton functionality."""
    results = TestResults()
    
    try:
        # Test singleton pattern
        model1 = SharedNeRFModel()
        model2 = SharedNeRFModel()
        
        if model1 is model2:
            results.record("shared_model_singleton", True)
        else:
            results.record("shared_model_singleton", False, "Not a proper singleton")
        
        # Test model loading (with fake checkpoint)
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_file:
            # Create fake checkpoint
            fake_checkpoint = {
                'coarse_model': NeRFModel().state_dict(),
                'fine_model': NeRFModel().state_dict()
            }
            torch.save(fake_checkpoint, tmp_file.name)
            
            try:
                model1.load_models(tmp_file.name, 'cpu')
                coarse, fine = model1.get_models()
                
                if coarse is not None and fine is not None:
                    results.record("shared_model_loading", True)
                else:
                    results.record("shared_model_loading", False, "Models not loaded properly")
                    
            except Exception as e:
                results.record("shared_model_loading", False, str(e))
    
    except Exception as e:
        results.record("shared_model_basic", False, str(e))
    
    return results


def test_device_detection():
    """Test device detection and compatibility."""
    results = TestResults()
    
    try:
        # Test MPS detection
        mps_available = torch.backends.mps.is_available()
        results.record("mps_detection", True, f"MPS available: {mps_available}")
        
        # Test CUDA detection
        cuda_available = torch.cuda.is_available()
        results.record("cuda_detection", True, f"CUDA available: {cuda_available}")
        
        # Test tensor creation on available devices
        cpu_tensor = torch.randn(10, 3, device='cpu')
        results.record("cpu_tensor_creation", True)
        
        if mps_available:
            try:
                mps_tensor = torch.randn(10, 3, device='mps')
                results.record("mps_tensor_creation", True)
            except Exception as e:
                results.record("mps_tensor_creation", False, str(e))
        
        if cuda_available:
            try:
                cuda_tensor = torch.randn(10, 3, device='cuda')
                results.record("cuda_tensor_creation", True)
            except Exception as e:
                results.record("cuda_tensor_creation", False, str(e))
    
    except Exception as e:
        results.record("device_detection_basic", False, str(e))
    
    return results


def test_memory_efficiency():
    """Test memory usage is reasonable."""
    results = TestResults()
    
    try:
        import psutil
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create models and do some operations
        model = NeRFModel()
        renderer = VolumeRenderer('cpu')
        
        # Simulate some processing
        for _ in range(10):
            positions = torch.randn(1000, 3)
            directions = torch.randn(1000, 3)
            density, rgb = model(positions, directions)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        if memory_increase < 500:  # Less than 500MB increase
            results.record("memory_efficiency", True, f"Memory increase: {memory_increase:.1f}MB")
        else:
            results.record("memory_efficiency", False, f"High memory usage: {memory_increase:.1f}MB")
    
    except Exception as e:
        results.record("memory_efficiency", False, str(e))
    
    return results


def run_integration_test():
    """Test basic integration without full training."""
    results = TestResults()
    
    try:
        # Test end-to-end flow with minimal data
        from src.benchmark.pytorch_renderers import PyTorchCPURenderer
        
        # Create renderer
        renderer = PyTorchCPURenderer()
        
        # Create fake checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_file:
            fake_checkpoint = {
                'coarse_model': NeRFModel().state_dict(),
                'fine_model': NeRFModel().state_dict()
            }
            torch.save(fake_checkpoint, tmp_file.name)
            
            # Setup renderer
            renderer.setup(tmp_file.name)
            
            # Test rendering
            camera_pose = torch.eye(4, dtype=torch.float32)
            camera_pose[2, 3] = 4.0  # Move camera back
            
            rgb_img, depth_img = renderer.render_image(
                camera_pose, resolution=(64, 48), samples_per_ray=16
            )
            
            if rgb_img.shape == (48, 64, 3):
                results.record("integration_rgb_shape", True)
            else:
                results.record("integration_rgb_shape", False, f"Expected (48, 64, 3), got {rgb_img.shape}")
            
            if depth_img.shape == (48, 64):
                results.record("integration_depth_shape", True)
            else:
                results.record("integration_depth_shape", False, f"Expected (48, 64), got {depth_img.shape}")
    
    except Exception as e:
        results.record("integration_test", False, str(e))
    
    return results


def main():
    """Run all tests with clean workspace management."""
    print("🧪 Running NeRF System Tests")
    print("=" * 50)
    print("Testing core functionality with minimal workspace impact...")
    print()
    
    # Create temporary test output directory
    test_output_dir = Path("tests/outputs")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        all_results = []
        
        # Run test suites
        print("🔬 Testing Positional Encoding...")
        all_results.append(test_positional_encoding())
        
        print("\n🧠 Testing NeRF Model...")
        all_results.append(test_nerf_model())
        
        print("\n📐 Testing Volume Renderer...")
        all_results.append(test_volume_renderer())
        
        print("\n🔄 Testing Shared Model...")
        all_results.append(test_shared_model())
        
        print("\n💻 Testing Device Detection...")
        all_results.append(test_device_detection())
        
        print("\n🎯 Testing Memory Efficiency...")
        all_results.append(test_memory_efficiency())
        
        print("\n🔗 Testing Integration...")
        all_results.append(run_integration_test())
        
        # Aggregate results
        total_passed = sum(r.passed for r in all_results)
        total_failed = sum(r.failed for r in all_results)
        total_tests = total_passed + total_failed
        
        print("\n" + "=" * 50)
        print(f"📊 Overall Results: {total_passed}/{total_tests} tests passed")
        
        if total_failed == 0:
            print("✅ All tests passed! System is ready for use.")
        else:
            print(f"❌ {total_failed} tests failed. Review issues above.")
            
            # Show failed test summary
            print("\n🔍 Failed Test Summary:")
            for result in all_results:
                for test in result.tests:
                    if not test['passed']:
                        print(f"   • {test['name']}: {test['message']}")
        
        # Save test report
        report_path = test_output_dir / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"NeRF System Test Report\n")
            f.write(f"Total: {total_passed}/{total_tests} passed\n\n")
            
            for i, result in enumerate(all_results):
                f.write(f"Test Suite {i+1}:\n")
                for test in result.tests:
                    status = "PASS" if test['passed'] else "FAIL"
                    f.write(f"  {test['name']}: {status}\n")
                    if test['message']:
                        f.write(f"    {test['message']}\n")
                f.write("\n")
        
        print(f"\n📝 Test report saved to: {report_path}")
        
        return total_failed == 0
    
    finally:
        # Clean up test outputs (keep only report)
        try:
            for item in test_output_dir.iterdir():
                if item.name != "test_report.txt":
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
        except:
            pass  # Don't fail if cleanup fails


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
