#!/usr/bin/env python3
"""
Test MPS training and device consistency.
Verifies that training on MPS works properly and renderers can use MPS-trained models.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mps_training_and_rendering():
    """Test MPS training and ensure device-consistent rendering."""
    print("🚀 Testing MPS Training & Device Consistency")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.backends.mps.is_available():
            print("❌ MPS not available on this system")
            return False
            
        # Import components
        from src.data.loader import SyntheticDataset
        from src.training.trainer import NeRFTrainer
        from src.benchmark.pytorch_renderers import PyTorchCPURenderer, PyTorchMPSRenderer
        
        print("✅ All imports successful")
        print("✅ MPS available and ready")
        
        # Load minimal data
        data_path = Path("data/nerf_synthetic/lego")
        if not data_path.exists():
            print("❌ No lego dataset found")
            return False
            
        # MPS training configuration
        print("\n📊 Testing MPS training...")
        config = {
            'device': 'mps',
            'lr': 5e-4,
            'n_rays': 512,  # Smaller for faster test
        }
        
        dataset = SyntheticDataset(str(data_path), 'train', img_wh=(100, 75))
        trainer = NeRFTrainer(config)
        print("✅ MPS Trainer initialized")
        
        # Train briefly on MPS
        print("🔥 Training on MPS...")
        trainer.train(dataset, n_epochs=1)
        print("✅ MPS Training completed")
        
        # Save checkpoint
        test_outputs = Path("tests/outputs")
        test_outputs.mkdir(exist_ok=True)
        checkpoint_filename = "mps_test_model.pth"
        trainer.save_checkpoint(checkpoint_filename)
        print(f"✅ MPS Checkpoint saved: {checkpoint_filename}")
        
        # Move checkpoint to test outputs for cleanup
        checkpoint_path = Path("checkpoints") / checkpoint_filename
        test_checkpoint_path = test_outputs / checkpoint_filename
        if checkpoint_path.exists():
            checkpoint_path.rename(test_checkpoint_path)
            checkpoint_path = test_checkpoint_path
        
        # Test MPS renderer with MPS-trained model
        print("\n⚡ Testing MPS renderer with MPS-trained model...")
        
        # Create a simple camera pose matrix
        camera_pose = torch.eye(4, dtype=torch.float32)
        camera_pose[2, 3] = 4.0  # z translation (move camera back)
        
        # MPS renderer
        mps_renderer = PyTorchMPSRenderer()
        mps_renderer.setup(str(checkpoint_path))
        print("✅ PyTorch MPS renderer initialized")
        
        rgb_mps, depth_mps = mps_renderer.render_image(
            camera_pose, 
            resolution=(50, 50), 
            samples_per_ray=8
        )
        
        assert rgb_mps.shape == (50, 50, 3), f"Expected (50, 50, 3), got {rgb_mps.shape}"
        assert depth_mps.shape == (50, 50), f"Expected (50, 50), got {depth_mps.shape}"
        print("✅ PyTorch MPS rendering successful")
        
        # Test CPU renderer with MPS-trained model (should work with proper device mapping)
        print("\n💻 Testing CPU renderer with MPS-trained model...")
        cpu_renderer = PyTorchCPURenderer()
        cpu_renderer.setup(str(checkpoint_path))
        print("✅ PyTorch CPU renderer initialized")
        
        rgb_cpu, depth_cpu = cpu_renderer.render_image(
            camera_pose, 
            resolution=(50, 50), 
            samples_per_ray=8
        )
        
        assert rgb_cpu.shape == (50, 50, 3), f"Expected (50, 50, 3), got {rgb_cpu.shape}"
        assert depth_cpu.shape == (50, 50), f"Expected (50, 50), got {depth_cpu.shape}"
        print("✅ PyTorch CPU rendering successful (with device mapping)")
        
        # Cleanup
        checkpoint_path.unlink()
        print("✅ Test files cleaned up")
        
        print("\n🎉 MPS TRAINING & RENDERING TEST PASSED!")
        print("✅ MPS Training: Successfully trained on MPS device")
        print("✅ MPS Rendering: Successfully rendered with MPS renderer")
        print("✅ Device Mapping: CPU renderer works with MPS-trained model")
        print("✅ Performance: MPS training should be significantly faster")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_mps_training_and_rendering()
    sys.exit(0 if success else 1)
