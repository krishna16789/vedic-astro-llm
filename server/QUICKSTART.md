# Quick Start Guide

Get the Vedic Astrology LLM server running in 3 simple steps!

## 🚀 One-Command Setup

```bash
# From the project root directory
cd server && pip install -r requirements.txt && python app.py
```

That's it! The server will:
1. Install dependencies
2. Download Mistral-7B model (~14GB, one-time)
3. Start the server on http://localhost:5000

## ⏱️ What to Expect

### First-Time Setup (~10-15 minutes)
- **Installing dependencies**: 2-3 minutes
- **Downloading model**: 5-10 minutes (depending on internet speed)
- **Loading model**: 1-2 minutes

### Subsequent Runs (~2 minutes)
- Model is cached, only needs to load into memory

## 📊 Using the Chart Calculator (No Wait!)

While the model is downloading/loading, you can immediately use the Chart Calculator:

1. Open http://localhost:5001
2. Click "📊 Chart Calculator" tab
3. Enter birth details and get instant planetary positions

The Chart Calculator works independently - no model required!

## 💬 Chat Interface

Once you see:
```
✓ Model loaded successfully on cuda
```

The chat is ready! Ask questions like:
- "What does Sun in Aries mean in Vedic astrology?"
- "Explain the significance of Ashwini nakshatra"
- "What are planetary yogas?"

## 🎯 Performance Tips

### For Faster Inference:
- **Use GPU**: Automatic if CUDA is available
- **CPU Mode**: Works but slower (20-40s per response)
- **Chart Calculator**: Always instant!

### If Model Won't Load:
1. Check you have 16GB+ RAM (or 8GB+ VRAM for GPU)
2. Ensure 20GB+ free disk space
3. Use Chart Calculator while troubleshooting

## 📝 Example API Calls

### Calculate a Chart:
```bash
curl -X POST http://localhost:5001/api/calculate-chart \
  -H "Content-Type: application/json" \
  -d '{
    "datetime": "1990-01-01T10:30:00",
    "latitude": 28.6139,
    "longitude": 77.2090
  }'
```

### Chat with AI:
```bash
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Vedic astrology?",
    "history": []
  }'
```

## 🔧 Common Issues

### "Model is not currently loaded"
- Wait for model to finish downloading
- Check console for error messages
- Use Chart Calculator in the meantime

### Slow responses
- Normal for first response (10-30s)
- Faster on subsequent queries
- GPU dramatically improves speed

### Out of memory
- Close other applications
- Server will try CPU mode automatically
- Chart Calculator always works!

## 🎉 You're Ready!

The server is now running. While training your model in Colab, you can:
1. Test calculations with the Chart Calculator
2. Chat with Mistral-7B about Vedic astrology
3. Experiment with API endpoints
4. Prepare for when your fine-tuned model is ready!

Happy calculating! 🌟