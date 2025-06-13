# NeRF Training and Unified Benchmark System

A clean, maintainable implementation of Neural Radiance Fields (NeRF) with unified benchmarking across multiple execution methods, optimized for Apple M3 Pro.

## 🎯 Goal

Train a NeRF model on synthetic data, then benchmark different rendering execution methods using the **same trained model** to compare pure execution performance rather than algorithmic differences.

## 🏗️ Architecture

```
nerf-dbr/
├── src/
│   ├── models/          # NeRF network implementation
│   ├── data/            # Dataset loading utilities
│   ├── utils/           # Volume rendering utilities
│   ├── training/        # Training loop and optimization
│   └── benchmark/       # Unified benchmark system
├── data/                # Your datasets (synthetic, LLFF, 360°)
├── checkpoints/         # Saved model weights
├── sample_renders/      # Visual comparison outputs
└── main.py              # Main training and benchmark script
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Activate environment
source activate_nerf.sh
```

### 2. Train Model and Run Benchmark

```bash
# Full pipeline: train + benchmark (recommended)
python main.py --epochs 20

# Quick training for testing
python main.py --epochs 5

# Benchmark only (if you have a trained model)
python main.py --benchmark_only --checkpoint checkpoints/final_model.pth
```

## 📊 What Gets Benchmarked

The system compares **execution methods** using the same trained NeRF:

### Unified Renderers

- **PyTorch MPS** - Apple M3 Pro Metal Performance Shaders
- **PyTorch CPU** - Multi-threaded CPU execution
- **PyTorch CUDA** - NVIDIA GPU acceleration (if available)
- **NumPy + Numba** - JIT-compiled CPU execution

### Test Configurations

- **Resolutions**: 200×150, 400×300, 800×600
- **Samples per ray**: 32, 64, 128
- **Multiple camera views** for statistical accuracy

### Metrics Measured

- **Render time** per image
- **Memory usage** during rendering
- **Rays per second** throughput
- **Device utilization** efficiency

## 🔧 Key Features

### Clean Implementation

- **DRY Code**: No duplication, shared base classes
- **Well Documented**: Clear docstrings and comments
- **Minimal**: Focused on core functionality
- **Maintainable**: Modular design, easy to extend

### Unified Model Guarantee

- **Single Source of Truth**: All renderers use identical trained weights
- **Fair Comparison**: Same network architecture and parameters
- **Standardized Input**: Identical ray generation and sampling
- **Controlled Variables**: Only execution method differs

### M3 Pro Optimized

- **MPS Acceleration**: Leverages Apple Silicon GPU
- **Memory Efficient**: Chunked processing for large images
- **Device Detection**: Automatically uses best available hardware
- **Performance Monitoring**: Real-time memory and timing tracking

## 📈 Example Output

```
🏆 PERFORMANCE HIGHLIGHTS:
   Fastest Method: PyTorch MPS (15,420 rays/sec)
   Most Memory Efficient: NumPy + Numba (245.1 MB)
   Maximum Speedup vs CPU: 3.2x

💡 RECOMMENDATIONS:
   • For Development: Use PyTorch MPS for fastest iteration
   • For Production: Balance performance vs hardware availability
   • For Mobile/Edge: Consider memory constraints (NumPy + Numba)
```

### Generated Files

- `unified_benchmark_results.csv` - Detailed performance data
- `unified_performance_comparison.png` - Visual charts
- `sample_renders/` - Quality comparison images
- `training_losses.png` - Training progress

## 🎛️ Configuration

### Training Parameters

```python
config = {
    'device': 'mps',          # 'mps', 'cuda', or 'cpu'
    'lr': 5e-4,               # Learning rate
    'n_coarse': 64,           # Coarse samples per ray
    'n_fine': 128,            # Fine samples per ray
    'chunk_size': 1024,       # Ray processing chunk size
    'n_rays': 1024,           # Rays per training batch
}
```

### Benchmark Parameters

```python
resolutions = [(200, 150), (400, 300), (800, 600)]
samples_per_ray_options = [32, 64, 128]
n_views = 2  # Camera poses to test
```

## 📊 Understanding Results

### Performance Metrics

- **Rays/Second**: Higher is better (throughput)
- **Render Time**: Lower is better (latency)
- **Memory Usage**: Lower is better (efficiency)
- **Speedup vs CPU**: Acceleration factor

### Interpreting Charts

- **Bar Charts**: Compare methods across resolutions
- **Scatter Plot**: Efficiency (throughput vs memory)
- **Sample Renders**: Visual quality verification

## 🔍 Dataset Requirements

The system works with NeRF synthetic dataset format:

```
data/nerf_synthetic/lego/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── train/
│   ├── r_0.png
│   └── ...
├── val/
└── test/
```

Download from [NeRF official data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

## ⚡ Performance Tips

### For Fastest Training

- Use `--epochs 5` for quick testing
- Reduce `n_rays` in config for less memory
- Use MPS device on Apple Silicon

### For Best Benchmark Results

- Close other applications during benchmark
- Use consistent power settings
- Run multiple times and average results

### Memory Management

- Reduce `chunk_size` if running out of memory
- Lower resolution for memory-constrained systems
- Monitor memory usage in Activity Monitor

## 🛠️ Extending the System

### Adding New Renderers

```python
class MyCustomRenderer(BaseUnifiedRenderer):
    def __init__(self):
        super().__init__("My Custom Method", "cpu")

    def execute_volume_rendering(self, densities, colors, z_vals, ray_dirs):
        # Your custom volume rendering implementation
        pass

    def render_image(self, camera_pose, resolution, samples_per_ray):
        # Your custom image rendering implementation
        pass
```

### Adding New Metrics

Extend `BenchmarkResult` dataclass and modify `generate_report()` method.

## 🐛 Troubleshooting

### Common Issues

**MPS Not Available**

```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Import Errors**

```bash
# Ensure environment is activated
source activate_nerf.sh

# Reinstall if needed
pip install -r requirements.txt
```

**Out of Memory**

- Reduce `chunk_size` in config
- Use smaller resolutions
- Lower `samples_per_ray`

**Slow Performance**

- Check device utilization in Activity Monitor
- Ensure MPS is being used: look for "GPU" usage
- Close other applications

## 📚 Implementation Details

### Volume Rendering Equation

```
C(r) = ∑ Ti · (1 - exp(-σi·δi)) · ci
Ti = exp(-∑[j=1 to i-1] σj·δj)
```

### Network Architecture

- **Input**: Positional encoded (x,y,z) + viewing direction
- **Architecture**: 8-layer MLP with skip connection at layer 4
- **Output**: Volume density σ + RGB color

### Sampling Strategy

- **Coarse**: Uniform sampling along rays
- **Fine**: Same spacing (simplified, can be extended to importance sampling)

## 🔬 Research Applications

This system is designed for:

- **Performance Research**: Comparing execution methods
- **Hardware Optimization**: M3 Pro vs other platforms
- **Algorithm Development**: Testing new rendering techniques
- **Educational Use**: Understanding NeRF internals

## 📄 License

MIT License - Feel free to use for research and educational purposes.

## 🤝 Contributing

1. Follow the existing clean, documented style
2. Add tests for new functionality
3. Update README for new features
4. Keep workspace clean (no test artifacts)

---

Built with ❤️ for clean, maintainable NeRF research on Apple Silicon.
# nerf-dbr
