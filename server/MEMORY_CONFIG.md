# Memory Configuration Guide

The server now has memory-efficient settings to prevent system crashes. Configure these settings in [`app.py`](app.py) based on your system.

## Quick Configuration (Lines 38-42 in app.py)

```python
LOAD_MODEL_ON_STARTUP = False  # Set to False to prevent crashes
USE_4BIT_QUANTIZATION = True   # Reduces memory from ~14GB to ~4GB  
USE_8BIT_QUANTIZATION = False  # Alternative: ~7GB memory
MAX_MEMORY_GB = 8              # Maximum memory to use
```

## Memory Usage Comparison

| Configuration | GPU Memory | RAM (CPU mode) | Speed | Quality |
|--------------|-----------|----------------|-------|---------|
| Full (FP16) | ~14GB | ~28GB | Fast | Best |
| 8-bit | ~7GB | ~14GB | Medium | Very Good |
| 4-bit | ~4GB | ~8GB | Medium | Good |
| CPU Only | N/A | ~28GB | Slow | Best |

## Recommended Settings by System

### 💻 Limited RAM (8-16GB)
```python
LOAD_MODEL_ON_STARTUP = False  # Load on first request
USE_4BIT_QUANTIZATION = True
MAX_MEMORY_GB = 6
```
**Result**: ~4-6GB memory usage, slower but works

### 🖥️ Medium RAM (16-32GB) + GPU
```python
LOAD_MODEL_ON_STARTUP = True   # Load immediately
USE_4BIT_QUANTIZATION = True
MAX_MEMORY_GB = 8
```
**Result**: ~4-8GB memory usage, fast

### 🚀 High RAM (32GB+) + GPU
```python
LOAD_MODEL_ON_STARTUP = True
USE_4BIT_QUANTIZATION = False  # Use full precision
USE_8BIT_QUANTIZATION = False
MAX_MEMORY_GB = 12
```
**Result**: ~14GB memory usage, fastest, best quality

### 🐢 CPU Only (No GPU)
```python
LOAD_MODEL_ON_STARTUP = False  # Definitely don't load on startup
USE_4BIT_QUANTIZATION = False  # Quantization needs GPU
USE_8BIT_QUANTIZATION = False
MAX_MEMORY_GB = 16
```
**Result**: Very slow but works, uses ~28GB RAM

## What Each Setting Does

### LOAD_MODEL_ON_STARTUP
- `True`: Loads model when server starts (faster first response, uses memory immediately)
- `False`: Loads model on first chat request (slower first response, but server starts fast)
- **Recommendation**: Use `False` to prevent crashes, especially on limited RAM

### USE_4BIT_QUANTIZATION
- Reduces model size by 75% with minimal quality loss
- Requires: GPU + bitsandbytes library
- **Recommendation**: Always use `True` on GPU systems with <16GB VRAM

### USE_8BIT_QUANTIZATION  
- Reduces model size by 50%
- Better quality than 4-bit, more memory than 4-bit
- Requires: GPU + bitsandbytes library
- **Recommendation**: Use if you have 8-12GB VRAM and want better quality

### MAX_MEMORY_GB
- Limits how much GPU memory the model can use
- Helps prevent out-of-memory crashes
- **Recommendation**: Set to 1-2GB less than your total VRAM

## Installation Notes

### For Quantization (Highly Recommended)
```bash
pip install bitsandbytes
```

This enables 4-bit and 8-bit quantization. If installation fails:
- **Linux/WSL**: Usually works out of box
- **Windows**: May need Visual Studio C++ build tools
- **Mac**: Limited support, use CPU mode instead

## Troubleshooting

### "CUDA out of memory"
1. Reduce `MAX_MEMORY_GB` by 2
2. Enable `USE_4BIT_QUANTIZATION = True`
3. Set `LOAD_MODEL_ON_STARTUP = False`

### "System froze/crashed"
1. Set `LOAD_MODEL_ON_STARTUP = False`
2. Enable `USE_4BIT_QUANTIZATION = True`
3. Close other applications before loading model
4. Use Chart Calculator (always works, no model needed!)

### "Model loading very slow"
- First time: Downloads ~14GB, takes 5-10 minutes
- Subsequent times: Loads from cache, 1-2 minutes
- With quantization: Faster loading

### "bitsandbytes not found"
```bash
pip install bitsandbytes

# If that fails on Windows:
pip install bitsandbytes-windows

# If that fails, disable quantization:
USE_4BIT_QUANTIZATION = False
USE_8BIT_QUANTIZATION = False
```

## Testing Your Configuration

1. Start server: `python server/app.py`
2. Watch console for memory usage
3. If it loads successfully:
   ```
   ✓ Model loaded successfully on cuda
   ✓ GPU Memory allocated: 4.23 GB
   ```
4. If you see errors, reduce memory settings

## Alternative: Use Chart Calculator Only

If model loading keeps failing, you can still use the Chart Calculator:
1. The calculator works without loading any AI model
2. Provides instant planetary position calculations  
3. Perfect for testing while you configure memory settings

## Monitoring Memory

### On Linux/Mac:
```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Watch RAM
htop
```

### On Windows:
```powershell
# Task Manager -> Performance tab
# Or use: nvidia-smi for GPU
```

## Best Practice

Start conservative and increase gradually:
1. Start with 4-bit quantization
2. Test if it works
3. If stable, try 8-bit or full precision
4. Find the sweet spot for your system

Remember: The Chart Calculator always works regardless of model loading!