# Baseline Training Run - 500 Epochs

## Training Configuration

- **Start Time**: June 13, 2025 at 1:47 PM CDT
- **End Time**: June 14, 2025 at ~9:45 PM CDT
- **Total Duration**: ~32 hours
- **Dataset**: NeRF Synthetic - Lego scene
- **Device**: Apple Silicon MPS (Metal Performance Shaders)

## Training Parameters

```python
learning_rate = 5e-4
lr_decay_steps = 250000
lr_decay_rate = 0.1
ray_batch_size = 2048
samples_per_ray = 64
importance_samples = 128
hidden_units = 256
num_layers = 10
position_encoding_levels = 10
direction_encoding_levels = 4
```

## Performance Results

### Training Progression (50-epoch intervals)

- **Epoch 50**: PSNR 8.2, MSE 0.38, SSIM 0.31
- **Epoch 100**: PSNR 8.9, MSE 0.32, SSIM 0.33
- **Epoch 150**: PSNR 9.1, MSE 0.31, SSIM 0.34
- **Epoch 200**: PSNR 9.3, MSE 0.30, SSIM 0.35
- **Epoch 250**: PSNR 9.5, MSE 0.29, SSIM 0.36
- **Epoch 300**: PSNR 9.8, MSE 0.26, SSIM 0.39 ← **PEAK PERFORMANCE**
- **Epoch 350**: PSNR 4.33, MSE 0.37, SSIM 0.21 ← **CATASTROPHIC COLLAPSE**
- **Epoch 400**: PSNR 4.42, MSE 0.36, SSIM 0.21
- **Epoch 450**: PSNR 4.36, MSE 0.37, SSIM 0.20
- **Epoch 500**: PSNR 4.31, MSE 0.37, SSIM 0.21 ← **FINAL FAILURE STATE**

### Training Analysis

- **Phase 1 (50-300)**: Gradual improvement, promising trajectory
- **Phase 2 (300-350)**: Catastrophic performance collapse (-56% PSNR)
- **Phase 3 (350-500)**: Persistent failure state, no recovery

### Root Cause Assessment

1. **Learning Rate Instability**: Likely too high (5e-4) causing gradient explosion
2. **Optimization Failure**: Unable to recover from bad state
3. **Possible Architecture Issues**: Over-parameterized positional encoding

## Files in This Directory

- `quick_render_epoch_XX.png`: Visual assessment renders for each 50-epoch interval
- `training_losses.png`: Complete training loss curve plot
- `checkpoints_baseline/`: All model checkpoints from this training run
- `BASELINE_SUMMARY.md`: This summary document

## Research Value

This baseline run provides:

- **Complete negative result documentation** for publication
- **Evidence base** for parameter optimization strategy
- **Apple Silicon MPS training characteristics** documentation
- **Failure pattern analysis** for future optimization

## Next Steps

Based on this baseline analysis, optimized training parameters have been designed:

- Reduced learning rate (5e-4 → 1e-4)
- Gradient clipping (prevent explosion)
- Increased model capacity (256 → 384 hidden units)
- Reduced positional encoding levels (10 → 8)
- Enhanced stability measures

Target performance for optimized run: PSNR > 20 dB (vs 4.31 dB baseline failure)
