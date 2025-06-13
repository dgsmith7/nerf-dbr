#!/bin/bash

# NeRF Training and Unified Benchmark Setup Script
# Optimized for macOS M3 Pro

echo "🚀 Setting up NeRF Training and Unified Benchmark System"
echo "=" * 60

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is optimized for macOS. Some features may not work on other systems."
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "🔥 Installing PyTorch with Metal Performance Shaders (MPS) support..."
pip install torch torchvision torchaudio

# Verify MPS availability
echo "🔍 Checking MPS availability..."
python3 -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'MPS Built: {torch.backends.mps.is_built()}')"

# Install other requirements
echo "📋 Installing additional requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p checkpoints
mkdir -p sample_renders
mkdir -p logs

# Download sample data if needed (optional)
echo "📊 Checking for sample data..."
if [ ! -d "data/nerf_synthetic/lego" ]; then
    echo "ℹ️  Sample synthetic data not found."
    echo "   You can download NeRF synthetic data from:"
    echo "   https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
    echo "   Extract to data/nerf_synthetic/"
else
    echo "✅ Sample data found at data/nerf_synthetic/lego"
fi

# Create activation script
echo "📝 Creating activation script..."
cat > activate_nerf.sh << 'EOF'
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
EOF

chmod +x activate_nerf.sh

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numba
print('✅ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device count: {torch.cuda.device_count()} CUDA, {1 if torch.backends.mps.is_available() else 0} MPS')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Summary:"
echo "   • Virtual environment: ./venv/"
echo "   • PyTorch with MPS support installed"
echo "   • All dependencies ready"
echo "   • Project directories created"
echo ""
echo "🚀 To get started:"
echo "   1. source activate_nerf.sh"
echo "   2. python main.py --help"
echo ""
echo "💡 For quick demo (if you have sample data):"
echo "   python main.py --epochs 5"
echo ""
echo "🔧 For benchmark only:"
echo "   python main.py --benchmark_only --checkpoint path/to/model.pth"
