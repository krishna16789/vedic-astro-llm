# Vedic Astrology LLM Server

A Flask-based web server for interacting with a local Mistral 7B model and testing Vedic astrology astronomical calculations.

## Features

- 💬 **Chat Interface**: Interact with locally-hosted Mistral 7B for Vedic astrology consultations
- 📊 **Chart Calculator**: Calculate Vedic birth charts with planetary positions
- 🌟 **Real-time Calculations**: Accurate ephemeris calculations using Swiss Ephemeris
- 🎨 **Modern UI**: Clean, responsive web interface
- 🚀 **Local Model**: No API keys needed - runs entirely on your machine

## Quick Start

### 1. Install Dependencies

```bash
cd server
pip install -r requirements.txt
```

> **Note**: This will download PyTorch and Transformers. The installation may take several minutes.

### 2. Run the Server

```bash
python app.py
```

**First-time startup**: The server will automatically download the Mistral-7B model (~14GB). This happens once and may take 5-10 minutes depending on your internet connection.

The server will start on `http://localhost:5001`

### System Requirements

- **RAM**: 16GB minimum (for CPU inference), 8GB VRAM for GPU
- **Storage**: ~20GB free space for model
- **GPU**: Optional but recommended (CUDA-compatible GPU for faster inference)

## Usage

### Chat with AI

1. Open your browser to `http://localhost:5001`
2. Wait for the model to load (you'll see status in the browser)
3. Click on the "💬 Chat with AI" tab
4. Ask questions about Vedic astrology:
   - "What does Sun in Aries mean?"
   - "Explain the significance of Ashwini nakshatra"
   - "What are the major yogas in Vedic astrology?"
   - "How do I interpret Moon in Rohini nakshatra?"

**Note**: First response may take 10-30 seconds as the model generates text. Subsequent responses are faster.

### Calculate Birth Chart

1. Click on the "📊 Chart Calculator" tab
2. Enter birth details:
   - **Date & Time**: When the person was born
   - **Latitude**: Birth location latitude (e.g., 28.6139 for Delhi)
   - **Longitude**: Birth location longitude (e.g., 77.2090 for Delhi)
3. Click "Calculate Chart"
4. View instant results:
   - Ascendant (Lagna) position
   - All planetary positions with:
     - Sign placement
     - Degree in sign
     - Nakshatra and Pada
     - Planetary speed

The Chart Calculator works independently of the AI model and provides instant results!

## API Endpoints

### Health Check
```http
GET /api/health
```

Returns server status and availability of features.

### Chat with AI
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What is Vedic astrology?",
  "history": []
}
```

### Calculate Chart
```http
POST /api/calculate-chart
Content-Type: application/json

{
  "datetime": "1990-01-01T10:30:00",
  "latitude": 28.6139,
  "longitude": 77.2090
}
```

### Get Planet Position
```http
POST /api/planet-position
Content-Type: application/json

{
  "planet": "Sun",
  "datetime": "1990-01-01T10:30:00"
}
```

### Get Available Planets
```http
GET /api/available-planets
```

## Example Chart Calculation

**Input:**
- Date/Time: January 1, 1990, 10:30 AM
- Location: Delhi, India (28.6139°N, 77.2090°E)

**Output:**
```json
{
  "ascendant": {
    "degree": 285.42,
    "sign": "Capricorn",
    "degree_in_sign": 15.42
  },
  "planets": {
    "Sun": {
      "longitude": 256.78,
      "sign": "Sagittarius",
      "degree_in_sign": 16.78,
      "nakshatra": "Purva Ashadha",
      "pada": 2,
      "speed": 1.0189
    },
    ...
  }
}
```

## Architecture

```
server/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web UI
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technologies Used

- **Flask**: Web framework
- **Mistral AI**: LLM for astrological consultations
- **Swiss Ephemeris**: Astronomical calculations
- **Vanilla JS**: Frontend interface

## Testing the Calculations

The Chart Calculator tab works independently of the Mistral API and can be used to:

1. Verify planetary positions for any date/time/location
2. Test the Swiss Ephemeris integration
3. Generate birth chart data while training the LLM

## Troubleshooting

### Model Not Loading

**Symptom**: Chat shows "Model is not currently loaded" message

**Solutions**:
1. **Check console output** - Look for error messages during model loading
2. **Insufficient RAM** - Model requires ~16GB RAM for CPU or ~8GB VRAM for GPU
3. **Download interrupted** - Delete `~/.cache/huggingface/` and restart server
4. **Disk space** - Ensure you have ~20GB free space

**Quick test without model**: Use the Chart Calculator tab - it works independently!

### Slow Response Times

- **First response**: 10-30 seconds is normal
- **On CPU**: Expect 20-40 seconds per response
- **On GPU**: 5-10 seconds per response
- **Solution**: If too slow, use the Chart Calculator for instant results

### CUDA Out of Memory

If you see CUDA OOM errors:
1. The server will automatically fall back to CPU
2. Or reduce model precision in `app.py`:
```python
torch_dtype=torch.float16  # Change to torch.float32 for CPU
```

### Port Already in Use

If port 5000 is busy, modify the last line in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change port
```

### Import Errors

Make sure you're in the project root directory when running the server:
```bash
cd /path/to/vedic-astro-llm
python server/app.py
```

### Model Downloads Keep Failing

Hugging Face downloads can be resumed. If interrupted:
1. Just restart the server - it will resume from where it stopped
2. Or manually download: `huggingface-cli download mistralai/Mistral-7B-v0.1`

## Development

### Adding New Endpoints

Add new routes in [`app.py`](app.py):

```python
@app.route('/api/your-endpoint', methods=['POST'])
def your_endpoint():
    data = request.json
    # Your logic here
    return jsonify({'result': 'data'})
```

### Modifying the UI

Edit [`templates/index.html`](templates/index.html) to customize the interface.

## Next Steps

- [ ] Add authentication for API access
- [ ] Implement caching for chart calculations
- [ ] Add more Vedic astrology calculation features
- [ ] Create API documentation with Swagger
- [ ] Add database for storing user queries and charts

## License

Part of the Vedic Astrology LLM project.