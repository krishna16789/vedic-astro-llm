# Server Setup - Complete Guide

## 🎯 What You Get

A complete web server for Vedic Astrology with:
- **AI Chat**: Talk to Mistral 7B about Vedic astrology
- **Chart Calculator**: Instant planetary position calculations
- **Clean UI**: Modern, responsive web interface

## ⚡ Quick Start (Safe Mode)

The server is now configured to **prevent crashes** with memory-efficient settings:

```bash
cd server
pip install -r requirements.txt
python app.py
```

**What happens**:
1. Server starts immediately (no crash!)
2. Chart Calculator works right away
3. Model loads only when you send first chat message

## 📊 Current Configuration

Check [`app.py`](app.py:38-42) for these settings:

```python
LOAD_MODEL_ON_STARTUP = False  # ✓ Prevents crashes
USE_4BIT_QUANTIZATION = True   # ✓ Uses only 4GB instead of 14GB
MAX_MEMORY_GB = 8              # ✓ Safety limit
```

## 🚀 What To Do Now

### Option 1: Use Chart Calculator Immediately ⭐
1. Start server: `python app.py`
2. Open http://localhost:5001
3. Click "📊 Chart Calculator"  
4. Enter birth details and get instant results!

**No model loading needed** - works right away while training runs!

### Option 2: Chat with AI (First Request Loads Model)
1. Start server
2. Open http://localhost:5001  
3. Send a chat message
4. Model loads (takes 2-5 minutes first time)
5. Get AI responses about Vedic astrology

## 💾 Memory Usage

| Feature | Memory Needed | Speed |
|---------|--------------|-------|
| Chart Calculator | <100MB | Instant |
| AI Chat (4-bit) | ~4GB | 5-10s per response |
| AI Chat (full) | ~14GB | 2-5s per response |

## 🔧 Adjust Settings

See [`MEMORY_CONFIG.md`](MEMORY_CONFIG.md) for detailed configuration options.

**Common adjustments**:

### If you have 16GB+ RAM and GPU:
```python
USE_4BIT_QUANTIZATION = True  # Fast + efficient
LOAD_MODEL_ON_STARTUP = True  # Load immediately
```

### If you have 8GB RAM:
```python
USE_4BIT_QUANTIZATION = True  # Essential!  
LOAD_MODEL_ON_STARTUP = False # Load on demand
MAX_MEMORY_GB = 6             # Be conservative
```

### If crashes still occur:
Just use the Chart Calculator! It always works and provides:
- Planetary positions
- Nakshatras and Padas
- Ascendant calculations
- All without AI model

## 📁 Project Structure

```
server/
├── app.py                 # Main server (configure here)
├── templates/
│   └── index.html        # Web UI
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── QUICKSTART.md         # Quick setup guide
├── MEMORY_CONFIG.md      # Memory optimization guide
└── SERVER_SETUP.md       # This file
```

## 🎓 Documentation

- **[README.md](README.md)** - Complete API documentation
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[MEMORY_CONFIG.md](MEMORY_CONFIG.md)** - Memory optimization
- **[SERVER_SETUP.md](SERVER_SETUP.md)** - This overview

## ✅ What Works Right Now

Even without the AI model loaded:

1. **Chart Calculator** ✓
   - Calculate birth charts
   - Get planetary positions
   - Find nakshatras and padas
   - Determine ascendant

2. **API Endpoints** ✓
   - `/api/calculate-chart`
   - `/api/planet-position`
   - `/api/available-planets`
   - `/api/health`

3. **Web Interface** ✓
   - Clean, modern UI
   - Responsive design
   - Real-time calculations

## 🚦 Server Status Indicators

When you open http://localhost:5000, you'll see:

- 🟢 **Mistral 7B: Loaded (cuda/cpu)** - AI chat ready
- 🟡 **Mistral 7B: Loading...** - AI loading, calculator works
- 🟢 **Calculations: Ready** - Always ready!

## 💡 Pro Tips

1. **While training in Colab**: Use Chart Calculator to test calculations
2. **Limited RAM**: Keep `LOAD_MODEL_ON_STARTUP = False`
3. **First chat slow**: Model loads on first request (expected)
4. **Always works**: Chart Calculator never needs model

## 🔄 During Training

While your model trains in Colab:
- ✅ Server runs locally
- ✅ Chart Calculator works  
- ✅ Test calculations
- ✅ Experiment with UI
- ✅ Prepare API integration

## 📞 Quick Reference

**Start server:**
```bash
python server/app.py
```

**Access UI:**
```
http://localhost:5000
```

**Test API:**
```bash
curl http://localhost:5000/api/health
```

**Calculate chart:**
```bash
curl -X POST http://localhost:5000/api/calculate-chart \
  -H "Content-Type: application/json" \
  -d '{"datetime":"1990-01-01T10:30:00","latitude":28.6139,"longitude":77.2090}'
```

## 🎉 You're All Set!

The server is configured for safe, memory-efficient operation. Start with the Chart Calculator and explore AI chat when ready!