#!/usr/bin/env python3
"""
Quick smoke test to verify basic Python functionality before full setup.
Tests that don't require external dependencies.
"""

import sys
import os
from pathlib import Path


def test_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Requires Python 3.8+")
        return False


def test_file_structure():
    """Verify expected file structure exists."""
    expected_files = [
        "src/models/nerf.py",
        "src/data/loader.py", 
        "src/utils/rendering.py",
        "src/training/trainer.py",
        "src/benchmark/base_renderer.py",
        "main.py",
        "requirements.txt",
        "setup.sh"
    ]
    
    missing = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if not missing:
        print(f"✅ File structure - All {len(expected_files)} expected files present")
        return True
    else:
        print(f"❌ File structure - Missing files: {missing}")
        return False


def test_import_structure():
    """Test that Python can parse our module structure."""
    try:
        # Test basic Python syntax parsing
        import ast
        
        test_files = [
            "src/models/nerf.py",
            "src/data/loader.py",
            "src/utils/rendering.py", 
            "main.py"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    try:
                        ast.parse(f.read())
                    except SyntaxError as e:
                        print(f"❌ Syntax error in {file_path}: {e}")
                        return False
        
        print("✅ Python syntax - All files parse correctly")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_data_availability():
    """Check if sample data is available."""
    data_paths = [
        "data/nerf_synthetic/lego",
        "data/nerf_llff_data/fern",
        "data/nerf_real_360"
    ]
    
    available = []
    for path in data_paths:
        if Path(path).exists():
            available.append(path)
    
    if available:
        print(f"✅ Sample data - Found: {available}")
        return True
    else:
        print("ℹ️  No sample data found - will need to download for training")
        print("   Synthetic data: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1")
        return True  # Not a failure, just info


def test_workspace_permissions():
    """Test that we can create files in workspace."""
    try:
        test_dir = Path("tests")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "write_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        print("✅ Workspace permissions - Can create files")
        return True
        
    except Exception as e:
        print(f"❌ Workspace permissions - Cannot create files: {e}")
        return False


def main():
    """Run smoke tests."""
    print("🔍 Running Pre-Setup Smoke Tests")
    print("=" * 40)
    print("These tests verify basic functionality before full setup...\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("File Structure", test_file_structure), 
        ("Python Syntax", test_import_structure),
        ("Data Availability", test_data_availability),
        ("Workspace Permissions", test_workspace_permissions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Smoke Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All smoke tests passed!")
        print("🚀 Ready to run: ./setup.sh")
        return True
    else:
        print("❌ Some smoke tests failed.")
        print("🔧 Please fix issues before running setup.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
