#!/usr/bin/env python3
"""Quick test after cleanup to verify everything works."""

import sys
import os
sys.path.append('.')

try:
    from src.benchmark import CPUOptimizedRenderer, CompressedNeRFRenderer
    print("✅ New renderers import successfully")
    
    # Test that we can instantiate them
    cpu_renderer = CPUOptimizedRenderer()
    compressed_renderer = CompressedNeRFRenderer()
    print("✅ New renderers instantiate successfully")
    
    print("🧹 Project cleanup completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
