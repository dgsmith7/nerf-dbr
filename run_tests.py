#!/usr/bin/env python3
"""
Coordinated test runner for NeRF system.
Runs tests in logical order with clean workspace management.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Command: {cmd}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run coordinated test suite."""
    print("🧪 NeRF System - Coordinated Test Suite")
    print("=" * 50)
    print("This runs all tests in logical order:")
    print("1. Smoke tests (no dependencies)")
    print("2. Setup verification")
    print("3. Unit tests (after setup)")
    print("4. Integration test (with real data)")
    print()
    
    # Track test results
    tests_run = []
    
    # Phase 1: Smoke tests (no dependencies required)
    print("📋 PHASE 1: PRE-SETUP SMOKE TESTS")
    print("=" * 50)
    
    smoke_success = run_command("python3 smoke_test.py", "Pre-setup smoke tests")
    tests_run.append(("Smoke Tests", smoke_success))
    
    if not smoke_success:
        print("\n❌ Smoke tests failed. Please fix basic issues before proceeding.")
        return False
    
    # Phase 2: Setup verification
    print("\n📦 PHASE 2: SETUP VERIFICATION")
    print("=" * 50)
    print("Checking if environment is set up...")
    
    venv_exists = Path("venv").exists()
    activate_script = Path("activate_nerf.sh").exists()
    
    if not venv_exists or not activate_script:
        print("⚠️  Environment not set up. Running setup...")
        setup_success = run_command("./setup.sh", "Environment setup")
        tests_run.append(("Environment Setup", setup_success))
        
        if not setup_success:
            print("\n❌ Setup failed. Cannot proceed with remaining tests.")
            return False
    else:
        print("✅ Environment already set up")
        tests_run.append(("Environment Setup", True))
    
    # Phase 3: Unit tests (requires dependencies)
    print("\n🔬 PHASE 3: UNIT TESTS")
    print("=" * 50)
    
    # Activate environment and run unit tests
    unit_cmd = "source venv/bin/activate && python test_system.py"
    unit_success = run_command(unit_cmd, "Unit tests")
    tests_run.append(("Unit Tests", unit_success))
    
    # Phase 4: Integration test (requires data)
    print("\n🔗 PHASE 4: INTEGRATION TEST")
    print("=" * 50)
    
    # Check if data is available
    lego_path = Path("data/nerf_synthetic/lego")
    if lego_path.exists():
        integration_cmd = "source venv/bin/activate && python test_integration.py"
        integration_success = run_command(integration_cmd, "Integration test with real data")
        tests_run.append(("Integration Test", integration_success))
    else:
        print("ℹ️  Lego dataset not found - skipping integration test")
        print("   Download from: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1")
        tests_run.append(("Integration Test", "SKIPPED"))
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in tests_run:
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = f"⏭️  {result}"
            skipped += 1
        
        print(f"  {test_name}: {status}")
    
    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 ALL TESTS SUCCESSFUL!")
        print("\nSystem is ready for production use:")
        print("  python main.py --epochs 20    # Full training and benchmark")
        print("  python main.py --help         # See all options")
        
        if skipped > 0:
            print(f"\nNote: {skipped} tests were skipped (missing data/optional features)")
        
        return True
    else:
        print(f"\n❌ {failed} TESTS FAILED")
        print("Please review the errors above and fix issues before using the system.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
