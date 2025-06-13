# NeRF Deep Benchmarking & Rendering (NeRF-DBR)

A comprehensive Neural Radiance Fields (NeRF) implementation with unified benchmarking across multiple execution methods. Compare performance of different rendering approaches using the **same trained model**, optimized for Apple Silicon and modern hardware.

## 🎯 Project Overview

Train a NeRF model once, then benchmark different execution methods to compare pure performance rather than algorithmic differences. This provides insights into hardware utilization, memory efficiency, and deployment optimization strategies.

## 💻 Development Hardware

This project was developed and tested on:

- **MacBook Pro** 14-Inch, Nov 2023
- **Chip** Apple M3 Pro
- **Memory** 36 GB
- **macOS** 15.5 (24F74)

_Performance benchmarks and optimizations are specifically tuned for Apple Silicon, though the system works cross-platform._

## ✨ Key Features

- **🧠 Multiple Execution Methods** - PyTorch (MPS/CPU/CUDA), NumPy+Numba, CPU Optimized, Compressed NeRF
- **🚀 Apple Silicon Optimized** - Automatic MPS acceleration with significant speedups
- **🔄 Unified Benchmarking** - Same trained model across all methods for fair comparison
- **📊 Comprehensive Analysis** - Performance metrics, visual renders, and memory profiling
- **🧹 Clean Architecture** - DRY principles, modular design, complete test coverage

## 🏗️ Architecture

```
nerf-dbr/
├── src/
│   ├── models/          # NeRF network implementation
│   ├── data/            # Dataset loading utilities
│   ├── training/        # Training loop and optimization
│   ├── utils/           # Volume rendering utilities
│   └── benchmark/       # Unified benchmark system
│       ├── pytorch_renderers.py    # MPS/CPU/CUDA implementations
│       ├── numpy_renderer.py       # NumPy + Numba implementation
│       ├── cpu_optimized_renderer.py   # Pure CPU optimized renderer
│       └── compressed_renderer.py      # Memory-efficient compressed renderer
├── data/                # Datasets (synthetic, LLFF, 360°)
├── checkpoints/         # Saved model weights
├── outputs/             # Benchmark results and sample renders
└── main.py              # Main training and benchmark script
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **macOS** (for MPS acceleration) or **Linux/Windows** (CPU/CUDA)
- **8GB+ RAM** recommended
- **Apple Silicon** for optimal performance

### 1. Setup Environment

```bash
# Clone repository (if from git)
git clone <repository-url>
cd nerf-dbr

# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate environment
source activate_nerf.sh
```

### 2. Get Dataset (if not included)

```bash
# Download official NeRF synthetic dataset
# Place in data/nerf_synthetic/lego/
# Or use your own NeRF-format dataset
```

### 3. Train and Benchmark

```bash
# Full pipeline: train + benchmark (recommended)
python main.py --epochs 50

# Quick training for testing
python main.py --epochs 5

# Benchmark only (if you have a trained model)
python main.py --benchmark_only

# Help and options
python main.py --help
```

### 4. Run Tests

```bash
# Complete test suite
python run_tests.py

# Individual tests
python test_system.py        # Unit tests
python test_integration.py   # Integration test
python smoke_test.py         # Pre-setup verification
```

## 📊 Benchmark Methods

The system compares these execution approaches using identical trained models:

### Available Renderers

| Method              | Description                    | Best For                   |
| ------------------- | ------------------------------ | -------------------------- |
| **PyTorch MPS**     | Apple Silicon GPU acceleration | Apple hardware development |
| **PyTorch CPU**     | Multi-threaded CPU execution   | Cross-platform baseline    |
| **PyTorch CUDA**    | NVIDIA GPU acceleration        | CUDA-enabled systems       |
| **NumPy + Numba**   | JIT-compiled CPU execution     | Lightweight deployment     |
| **CPU Optimized**   | Pure NumPy ray marching        | Maximum CPU efficiency     |
| **Compressed NeRF** | Quantized/pruned networks      | Edge/mobile deployment     |

### Performance Metrics

- **Render Time** - Latency per image (lower = better)
- **Rays/Second** - Throughput (higher = better)
- **Memory Usage** - Peak memory consumption
- **Visual Quality** - Saved sample renders for comparison

### Test Configurations

- **Resolutions**: 200×150, 400×300, 800×600
- **Samples per ray**: 32, 64, 128
- **Multiple camera views** for statistical accuracy

## � Output Files

All benchmark results and renders are saved to the `outputs/` directory:

```
outputs/
├── benchmark_results.csv           # Detailed performance metrics
├── performance_comparison.png      # Performance charts
└── sample_renders/                # Visual quality comparison
    ├── PyTorch_MPS/
    │   ├── view_0_rgb.png         # Color renders
    │   └── view_0_depth.png       # Depth maps
    ├── CPU_Optimized/
    ├── Compressed_NeRF/
    └── NumPy_Numba/
```

## 🎛️ Configuration

### Quick Start Options

```bash
# Full pipeline (recommended first run)
python main.py --epochs 50

# Quick test (5 minutes)
python main.py --epochs 5

# Benchmark only (reuse trained model)
python main.py --benchmark_only
```

### Epoch Recommendations

| Epochs | Time (M3 Pro) | Quality   | Use Case       |
| ------ | ------------- | --------- | -------------- |
| 5      | ~10 min       | Fair      | Quick testing  |
| 50     | ~1.5 hours    | Good      | Development    |
| 200    | ~6 hours      | Very Good | Demonstrations |
| 500+   | 15+ hours     | Excellent | Production     |

## �️ Troubleshooting

### Quick Fixes

**MPS not available**

```bash
# Check Apple Silicon support
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

**Memory issues**

```bash
# Reduce memory usage
python main.py --epochs 5  # Use fewer epochs for testing
```

**Import errors**

```bash
# Ensure environment is activated
source activate_nerf.sh
pip install -r requirements.txt
```

### Performance Tips

- **Close other applications** during benchmarking
- **Use adequate cooling** for sustained performance
- **Monitor Activity Monitor** for resource usage
- **Try different epoch counts** to balance quality vs time

## 📚 Technical Details

### Network Architecture

- **Input**: Positional encoded (x,y,z) + viewing direction
- **Architecture**: 8-layer MLP with skip connection at layer 4
- **Output**: Volume density σ + RGB color

### Volume Rendering

```
C(r) = ∑ Ti · (1 - exp(-σi·δi)) · ci
Ti = exp(-∑[j=1 to i-1] σj·δj)
```

### Compression Techniques (Compressed NeRF)

- **Quantization**: 8-bit/16-bit weight precision
- **Pruning**: Remove smallest magnitude weights
- **Mixed Precision**: Float16/32 computation balance

## 🔬 Research Applications

- **Performance Analysis**: Compare execution methods across hardware
- **Hardware Optimization**: Evaluate Apple Silicon vs traditional platforms
- **Memory Efficiency**: Test compression techniques for edge deployment
- **Algorithm Development**: Benchmark new rendering approaches

## � Testing

```bash
# Complete test suite
python run_tests.py

# Individual tests
python test_system.py        # Unit tests
python test_integration.py   # Integration tests
python smoke_test.py         # Quick verification
```

## 📄 License & Contributing

**MIT License** - Free for research and educational use.

**Contributing Guidelines:**

1. Follow clean, documented coding style
2. Add tests for new functionality
3. Update README for new features
4. Run full test suite: `python run_tests.py`

---

**Built with ❤️ for clean, maintainable NeRF research on modern hardware.**

_Updated: June 13, 2025 - Now with CPU Optimized and Compressed NeRF renderers_
