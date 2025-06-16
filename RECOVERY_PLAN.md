# Recovery Plan: GLSL NeRF Renderer

## Immediate Actions for Tomorrow

### 1. Clean Slate Setup

```bash
# Start fresh with clean branch
git checkout main
git pull
git checkout -b glsl-renderer-v2-clean
```

### 2. New File Structure (CRITICAL)

Create separate files to avoid string corruption:

```
src/benchmark/
├── glsl_renderer.py           # Python class only - NO GLSL CODE
├── shaders/
│   ├── vertex.glsl           # Simple vertex shader
│   ├── fragment_header.glsl   # Header with uniforms/defines
│   ├── debug_functions.glsl   # All debug mode functions
│   ├── mlp_functions.glsl     # MLP computation functions
│   └── fragment_main.glsl     # Main function with mode switching
```

### 3. Restore Known Working Components

Priority order:

1. **OpenGL Context**: `moderngl.create_context(standalone=True)`
2. **Weight Texture Upload**: Copy from working debug scripts
3. **Coordinate Debug Mode**: Simplest validation
4. **Positional Encoding**: We had this working perfectly
5. **Layer 0 MLP**: Achieved 1-8% error - we know the exact implementation
6. **Layer 1 Debug**: Focus on the array indexing issue

### 4. Validation Strategy

- Run precision tests after each component
- Compare against PyTorch at every step
- Maintain debug mode for each layer
- Create checkpoint commits frequently

## Key Implementation Notes

### Weight Texture Upload (WORKING CODE)

```python
# From debug_texture_sampling.py - this worked perfectly
def upload_weight_texture(ctx, weight_tensor):
    weight_np = weight_tensor.detach().cpu().numpy().astype(np.float32)
    if len(weight_np.shape) == 1:
        # Bias vector
        width = weight_np.shape[0]
        texture = ctx.texture((width, 1), 1, weight_np.tobytes())
    else:
        # Weight matrix
        height, width = weight_np.shape
        texture = ctx.texture((width, height), 1, weight_np.tobytes())
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    return texture
```

### Layer 0 MLP (VALIDATED WORKING)

The GLSL code for layer 0 was working with 1-8% error:

- Matrix multiplication with proper indexing
- Bias addition
- ReLU activation
- Full 256-neuron output validation

### Layer 1 Issue (SPECIFIC BUG)

The problem was in the array indexing/assignment:

- Layer 0 outputs were correct
- Weight sampling was correct
- But neurons 0 and 5 were returning identical values
- Likely a loop variable or array assignment bug

## Confidence Factors

### What We KNOW Works ✅

- Overall architecture and approach
- Texture-based weight storage
- Positional encoding implementation
- Layer 0 MLP computation
- Debug mode infrastructure
- Precision validation methodology

### What We Need to Debug 🔍

- Layer 1 array indexing (specific, isolated issue)
- GLSL compilation with separate files
- File loading and shader assembly

## Success Metrics

### Immediate Goals (Day 1)

- [ ] Clean file structure implemented
- [ ] OpenGL context working
- [ ] Weight textures uploaded
- [ ] Coordinate debug mode working
- [ ] Positional encoding restored (< 3% error)

### Short-term Goals (Days 2-3)

- [ ] Layer 0 MLP restored (< 8% error)
- [ ] Layer 1 array indexing bug fixed
- [ ] Layer 1 precision validation (< 5% error)
- [ ] Full precision test suite running

### Medium-term Goals (Week 1)

- [ ] All MLP layers implemented
- [ ] Ray marching and volume rendering
- [ ] Production benchmarking integration
- [ ] Performance optimization

---

**The methodical approach was working perfectly. We just need better file organization to prevent corruption.**
