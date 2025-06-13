#!/bin/bash
echo "🔥 Activating NeRF environment..."
source venv/bin/activate
echo "✅ Environment activated. MPS support available."
echo ""
echo "Quick start:"
echo "  python main.py --help                    # Show all options"
echo "  python main.py --epochs 10               # Quick training (10 epochs)"
echo "  python main.py --benchmark_only          # Benchmark only (skip training)"
echo ""
