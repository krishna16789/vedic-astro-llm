"""
Vedic Astrology LLM Server
Flask API server for model interaction and calculations
Updated with divisional chart support
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from calculations.core.ephemeris import VedicEphemeris

# Try to import transformers for local model loading
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Chat will use mock responses.")

app = Flask(__name__)
CORS(app)

# Initialize ephemeris calculator
ephemeris = VedicEphemeris()

# Initialize local Mistral model
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_loaded = False

# Configuration options - adjust these based on your system
LOAD_MODEL_ON_STARTUP = False  # Set to False to prevent crashes - load on first request instead
USE_4BIT_QUANTIZATION = True   # Reduces memory from ~14GB to ~4GB
USE_8BIT_QUANTIZATION = False  # Alternative: ~7GB memory
MAX_MEMORY_GB = 8              # Maximum memory to use (adjust based on your system)

def load_local_model():
    """Load local Mistral 7B model with memory-efficient settings"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        print("✓ Model already loaded")
        return True
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  Transformers not available. Install with: pip install transformers torch")
        return False
    
    try:
        model_name = "mistralai/Mistral-7B-v0.1"
        print(f"\n{'=' * 60}")
        print(f"Loading {model_name} on {device}...")
        print(f"Memory settings:")
        print(f"  - 4-bit quantization: {USE_4BIT_QUANTIZATION}")
        print(f"  - 8-bit quantization: {USE_8BIT_QUANTIZATION}")
        print(f"  - Max memory: {MAX_MEMORY_GB}GB")
        print(f"{'=' * 60}\n")
        
        # Load tokenizer (minimal memory)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded")
        
        # Prepare model loading arguments with memory optimizations
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add quantization if requested (requires bitsandbytes)
        if USE_4BIT_QUANTIZATION and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                print("Using 4-bit quantization (saves ~10GB memory)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("⚠️  bitsandbytes not installed, quantization disabled")
                print("   Install with: pip install bitsandbytes")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        elif USE_8BIT_QUANTIZATION and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                print("Using 8-bit quantization (saves ~7GB memory)")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("⚠️  bitsandbytes not installed, quantization disabled")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: f"{MAX_MEMORY_GB}GB"}
        else:
            # CPU mode - use float32 but with memory constraints
            model_kwargs["torch_dtype"] = torch.float32
            print(f"⚠️  Loading on CPU - this will be slow!")
            print(f"   Consider using GPU or reducing MAX_MEMORY_GB")
        
        # Load model
        print("Loading model... (this may take 2-5 minutes)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        model.eval()  # Set to evaluation mode
        
        print(f"✓ Model loaded successfully on {device}")
        
        # Print memory stats if on CUDA
        if device == "cuda":
            print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"✓ GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Falling back to mock mode")
        print("\n💡 Tips to reduce memory usage:")
        print("   1. Set USE_4BIT_QUANTIZATION = True in app.py")
        print("   2. Reduce MAX_MEMORY_GB")
        print("   3. Close other applications")
        print("   4. Use the Chart Calculator (always works!)\n")
        return False

# Conditionally load model on startup
if LOAD_MODEL_ON_STARTUP:
    print("\n" + "=" * 60)
    print("Initializing Mistral 7B Model on Startup...")
    print("=" * 60)
    model_loaded = load_local_model()
else:
    print("\n" + "=" * 60)
    print("Model Loading: DEFERRED")
    print("=" * 60)
    print("Model will load on first chat request (prevents crashes)")
    print("Use the Chart Calculator immediately!")
    print("=" * 60 + "\n")


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'ephemeris_available': True
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with local Mistral model
    Request: {"message": "user message", "history": [...]}
    """
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Load model on first request if not loaded yet
    if not model_loaded and TRANSFORMERS_AVAILABLE:
        print("\n🔄 Loading model on first request...")
        load_local_model()
    
    # If local model is loaded, use it
    if model is not None and tokenizer is not None:
        try:
            # Prepare the prompt with system message and conversation history
            system_prompt = """You are an expert Vedic astrologer with deep knowledge of:
- Planetary positions and their interpretations
- Nakshatras (lunar mansions) and their meanings
- Yogas and astrological combinations
- Birth chart analysis and predictions
- Vedic astrology principles and classical texts

Provide accurate, insightful answers based on Vedic astrology principles.
When discussing planetary positions, refer to the sidereal (Nirayana) zodiac.
Be respectful of the sacred nature of this knowledge."""

            # Build conversation context
            conversation = f"<s>[INST] {system_prompt}\n\n"
            
            # Add recent history (last 3 exchanges for context)
            for msg in history[-6:]:  # Last 3 user-assistant pairs
                if msg.get('role') == 'user':
                    conversation += f"{msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    conversation += f"[/INST] {msg.get('content', '')} </s><s>[INST] "
            
            # Add current user message
            conversation += f"{user_message} [/INST]"
            
            # Tokenize
            inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response (after the last [/INST])
            assistant_message = full_response.split("[/INST]")[-1].strip()
            
            return jsonify({
                'response': assistant_message,
                'model': 'mistral-7b-local',
                'device': device
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Model error: {str(e)}',
                'using_mock': True,
                'response': f"I encountered an error processing your request. Please try again or use the Chart Calculator for planetary position calculations."
            }), 500
    
    else:
        # Mock response when model is not loaded
        mock_response = f"""The Mistral 7B model is not currently loaded. This could be because:

1. The model is still loading (this takes 2-5 minutes on first start)
2. There was an error loading the model
3. Insufficient memory available

For now, I can help you with:
- Calculating planetary positions (use the Chart Calculator)
- Understanding Vedic astrology concepts
- Testing the ephemeris calculations

You asked: "{user_message}"

Try using the Chart Calculator tab to see real planetary calculations!"""
        
        return jsonify({
            'response': mock_response,
            'model': 'mock',
            'using_mock': True
        })


@app.route('/api/calculate-chart', methods=['POST'])
def calculate_chart():
    """
    Calculate Vedic astrology chart
    Request: {
        "datetime": "2023-01-01T10:30:00",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone_offset": 5.5  // Optional: hours offset from UTC (e.g., IST = 5.5)
    }
    
    Note: Swiss Ephemeris requires UTC time. If timezone_offset is provided,
    the datetime will be converted from local time to UTC automatically.
    """
    data = request.json
    
    try:
        # Parse datetime
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset (hours from UTC, e.g., IST = 5.5)
        timezone_offset = data.get('timezone_offset', 0)
        
        # Convert local time to UTC
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        # Get coordinates
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        if lat == 0 and lon == 0:
            return jsonify({'error': 'Valid latitude and longitude required'}), 400
        
        # Calculate chart using UTC time
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        return jsonify({
            'success': True,
            'chart': chart,
            'timezone_info': {
                'local_time': dt_local.isoformat(),
                'utc_time': dt_utc.isoformat(),
                'offset_hours': timezone_offset
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500


@app.route('/api/planet-position', methods=['POST'])
def planet_position():
    """
    Get specific planet position
    Request: {
        "planet": "Sun",
        "datetime": "2023-01-01T10:30:00"
    }
    """
    data = request.json
    
    try:
        planet_name = data.get('planet')
        dt_str = data.get('datetime')
        
        if not planet_name or not dt_str:
            return jsonify({'error': 'planet and datetime are required'}), 400
        
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        jd = ephemeris.get_julian_day(dt)
        
        position = ephemeris.get_planet_position(planet_name, jd)
        
        return jsonify({
            'success': True,
            'planet': planet_name,
            'position': {
                'longitude': position.longitude,
                'latitude': position.latitude,
                'sign': position.sign,
                'degree_in_sign': position.degree_in_sign,
                'nakshatra': position.nakshatra,
                'pada': position.nakshatra_pada,
                'speed': position.speed
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500


@app.route('/api/available-planets', methods=['GET'])
def available_planets():
    """Get list of available planets"""
    return jsonify({
        'planets': list(ephemeris.PLANETS.keys())
    })


@app.route('/api/search-location', methods=['GET'])
def search_location():
    """
    Search for location coordinates by place name
    Uses Nominatim (OpenStreetMap) geocoding API
    """
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        import urllib.parse
        import urllib.request
        import json
        
        # Use Nominatim API (free, no API key needed)
        encoded_query = urllib.parse.quote(query)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=5"
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'VedicAstroLLM/1.0')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        
        # Format results
        locations = []
        for item in data:
            locations.append({
                'name': item.get('display_name', ''),
                'latitude': float(item.get('lat', 0)),
                'longitude': float(item.get('lon', 0)),
                'type': item.get('type', ''),
            })
        
        return jsonify({
            'success': True,
            'locations': locations
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Location search failed: {str(e)}',
            'locations': []
        }), 500


@app.route('/api/generate-rasi-chart', methods=['POST'])
def generate_rasi_chart():
    """
    Generate Rasi chart (D1) data for visualization
    South Indian style: Signs are in fixed positions, planets move through them
    """
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset and convert to UTC
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data using UTC time
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Get ascendant sign number (0-11) where 0=Aries, 1=Taurus, etc.
        ascendant_degree = chart['ascendant']['degree']
        ascendant_sign_num = int(ascendant_degree / 30)
        ascendant_sign = ephemeris.SIGNS[ascendant_sign_num]
        
        # Initialize all 12 zodiac signs (Aries=1 to Pisces=12)
        # In South Indian chart, signs are fixed positions
        signs = {}
        for i in range(12):
            sign_num = i  # 0=Aries, 1=Taurus, ... 11=Pisces
            signs[i+1] = {  # Sign numbers 1-12 for display
                'number': i + 1,
                'sign': ephemeris.SIGNS[sign_num],
                'sign_num': sign_num,
                'planets': [],
                'has_ascendant': sign_num == ascendant_sign_num
            }
        
        # Place each planet in its zodiac sign
        for planet_name, planet_data in chart['planets'].items():
            planet_sign_num = int(planet_data['longitude'] / 30)
            sign_key = planet_sign_num + 1  # Convert to 1-12
            
            signs[sign_key]['planets'].append({
                'name': planet_name,
                'degree': planet_data['degree_in_sign'],
                'longitude': planet_data['longitude'],
                'speed': planet_data['speed'],
                'is_retrograde': planet_data.get('is_retrograde', False),
                'is_combust': planet_data.get('is_combust', False)
            })
        
        return jsonify({
            'success': True,
            'chart_type': 'Rasi (D1) - South Indian',
            'ascendant': {
                'sign': ascendant_sign,
                'degree': chart['ascendant']['degree_in_sign'],
                'sign_num': ascendant_sign_num
            },
            'signs': signs  # Changed from 'houses' to 'signs'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Chart generation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate-divisional-chart', methods=['POST'])
def generate_divisional_chart():
    """
    Generate divisional chart (Varga) data for visualization
    Currently supports D1 (Rasi) and D9 (Navamsa)
    """
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        varga = data.get('varga', 'D9')
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset and convert to UTC
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get base chart data
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Calculate divisional chart positions
        ascendant_degree = chart['ascendant']['degree']
        divisional_ascendant = ephemeris.calculate_divisional_chart(ascendant_degree, varga)
        divisional_ascendant_sign_num = int(divisional_ascendant / 30)
        divisional_ascendant_sign = ephemeris.SIGNS[divisional_ascendant_sign_num]
        
        # Initialize all 12 zodiac signs
        signs = {}
        for i in range(12):
            sign_num = i
            signs[i+1] = {
                'number': i + 1,
                'sign': ephemeris.SIGNS[sign_num],
                'sign_num': sign_num,
                'planets': [],
                'has_ascendant': sign_num == divisional_ascendant_sign_num
            }
        
        # Place each planet in its divisional chart sign
        for planet_name, planet_data in chart['planets'].items():
            divisional_longitude = ephemeris.calculate_divisional_chart(planet_data['longitude'], varga)
            planet_sign_num = int(divisional_longitude / 30)
            sign_key = planet_sign_num + 1
            
            signs[sign_key]['planets'].append({
                'name': planet_name,
                'degree': divisional_longitude % 30,
                'longitude': divisional_longitude,
                'speed': planet_data['speed'],
                'is_retrograde': planet_data.get('is_retrograde', False),
                'is_combust': planet_data.get('is_combust', False)
            })
        
        return jsonify({
            'success': True,
            'chart_type': f'{varga} Chart',
            'ascendant': {
                'sign': divisional_ascendant_sign,
                'degree': divisional_ascendant % 30,
                'sign_num': divisional_ascendant_sign_num
            },
            'signs': signs
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Divisional chart generation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Vedic Astrology LLM Server")
    print("=" * 60)
    print(f"✓ Ephemeris calculations: Ready")
    print(f"✓ Mistral 7B Model: {'Loaded on ' + device if model_loaded else 'Not loaded (mock mode)'}")
    print(f"✓ Device: {device}")
    print("\nStarting server on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
