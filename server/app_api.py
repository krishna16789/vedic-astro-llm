"""
Vedic Astrology Server with Multiple API Support
Supports: Groq, Gemini 2.0 Flash, DeepSeek, OpenRouter
All calculations done locally in Python
Configuration via .env file
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import sys
from pathlib import Path
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from the server directory (where this file is located)
    env_path = Path(__file__).parent / '.env'
    # override=True forces .env to override system environment variables
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"✓ Loaded .env from: {env_path}")
    print(f"  API_PROVIDER: {os.getenv('API_PROVIDER', 'NOT SET')}")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("⚠️  Using system environment variables only")
except Exception as e:
    print(f"⚠️  Error loading .env: {e}")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from calculations.core.ephemeris import VedicEphemeris
from calculations.core.astro_tools import AstroTools

app = Flask(__name__)
CORS(app)

# Initialize calculators
ephemeris = VedicEphemeris()
astro_tools = AstroTools()

# Store last calculated chart
last_calculated_chart = None
last_chart_timestamp = None

# API Configuration - Load from .env file
API_PROVIDER = os.getenv('API_PROVIDER', 'gemini').lower()

# Load API keys from .env
API_KEYS = {
    'groq': os.getenv('GROQ_API_KEY', ''),
    'gemini': os.getenv('GEMINI_API_KEY', ''),
    'deepseek': os.getenv('DEEPSEEK_API_KEY', ''),
    'openrouter': os.getenv('OPENROUTER_API_KEY', '')
}

# Model configuration (can be overridden in .env)
MODEL_CONFIG = {
    'groq': os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
    'gemini': os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview'),  # Latest Gemini 2.0 Flash
    'deepseek': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
    'openrouter': os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-70b-instruct:free')
}

# Get API key for selected provider
API_KEY = API_KEYS.get(API_PROVIDER, '')
MODEL_NAME = MODEL_CONFIG.get(API_PROVIDER, '')

# Initialize based on provider
llm_client = None

if API_PROVIDER == 'groq':
    try:
        from groq import Groq
        if API_KEY:
            llm_client = Groq(api_key=API_KEY)
            print(f"✓ Groq API configured")
            print(f"  Model: {MODEL_NAME}")
        else:
            print(f"⚠️  GROQ_API_KEY not set in .env file")
    except ImportError:
        print("⚠️  Install groq: pip install groq")

elif API_PROVIDER == 'gemini':
    try:
        import google.generativeai as genai
        if API_KEY:
            genai.configure(api_key=API_KEY)
            llm_client = genai.GenerativeModel(MODEL_NAME)
            print(f"✓ Gemini API configured")
            print(f"  Model: {MODEL_NAME} (Latest Gemini 2.0 Flash)")
        else:
            print(f"⚠️  GEMINI_API_KEY not set in .env file")
    except ImportError:
        print("⚠️  Install google-generativeai: pip install google-generativeai")

elif API_PROVIDER == 'deepseek':
    try:
        from openai import OpenAI
        if API_KEY:
            llm_client = OpenAI(
                api_key=API_KEY,
                base_url="https://api.deepseek.com"
            )
            print(f"✓ DeepSeek API configured")
            print(f"  Model: {MODEL_NAME}")
        else:
            print(f"⚠️  DEEPSEEK_API_KEY not set in .env file")
    except ImportError:
        print("⚠️  Install openai: pip install openai")

elif API_PROVIDER == 'openrouter':
    try:
        from openai import OpenAI
        if API_KEY:
            llm_client = OpenAI(
                api_key=API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            print(f"✓ OpenRouter API configured")
            print(f"  Model: {MODEL_NAME}")
        else:
            print(f"⚠️  OPENROUTER_API_KEY not set in .env file")
    except ImportError:
        print("⚠️  Install openai: pip install openai")
else:
    print(f"⚠️  Unknown API_PROVIDER: {API_PROVIDER}")
    print(f"   Valid options: groq, gemini, deepseek, openrouter")


def call_llm(prompt: str) -> str:
    """Universal LLM caller - works with any provider"""
    
    if not llm_client:
        raise Exception("No API configured. Set API_KEY environment variable.")
    
    if API_PROVIDER == 'groq':
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    elif API_PROVIDER == 'gemini':
        response = llm_client.generate_content(prompt)
        return response.text
    
    elif API_PROVIDER in ['deepseek', 'openrouter']:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    raise Exception(f"Unknown API provider: {API_PROVIDER}")


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'api_provider': API_PROVIDER,
        'api_configured': llm_client is not None,
        'model': MODEL_NAME if llm_client else None,
        'tools_available': 9
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with LLM + local calculation tools"""
    data = request.json
    user_message = data.get('message', '')
    chart_data = data.get('chart_data', None)
    
    # Use last calculated chart if none provided
    if chart_data is None and last_calculated_chart is not None:
        chart_data = last_calculated_chart
        print(f"✓ Using last calculated chart from {last_chart_timestamp}")
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    if not llm_client:
        return jsonify({
            'error': 'API not configured',
            'response': f'API key not configured for {API_PROVIDER}.\n\n1. Copy server/.env.example to server/.env\n2. Add your {API_PROVIDER.upper()}_API_KEY in .env file\n3. Restart server\n\nOr set: export {API_PROVIDER.upper()}_API_KEY=your-key'
        }), 503
    
    try:
        # Build system prompt with chart data
        system_prompt = """You are an expert Vedic astrologer with deep knowledge of classical texts like BPHS, Jataka Parijata, and Phaladeepika.

CRITICAL INSTRUCTION: You MUST include detailed ASPECT ANALYSIS in every chart reading.

**ASPECT RULES (Drishti) - MANDATORY TO CHECK:**
1. All planets aspect the 7th house from their position (full 100% aspect)
2. Mars: Also aspects 4th and 8th houses from its position (special aspects)
3. Jupiter: Also aspects 5th and 9th houses from its position (special aspects)
4. Saturn: Also aspects 3rd and 10th houses from its position (special aspects)
5. Rahu/Ketu: Also aspect 5th and 9th houses from their position (like Jupiter)
6. Check for MUTUAL ASPECTS (when two planets aspect each other)

**YOUR ANALYSIS MUST INCLUDE:**

1. **PLANETARY ASPECTS (MANDATORY - Don't skip this!):**
   - Identify ALL significant aspects in the chart
   - Explain the INFLUENCE and EFFECTS of each major aspect
   - Note any mutual aspects (bidirectional influences)
   - Consider aspect strength based on dignity and house placement
   - Example: "Mars in 5th house aspects 8th house (transformation), 11th house (gains), and 12th house (losses)"

2. **YOGAS (Planetary Combinations):**
   - Raja Yogas (trikona + kendra lord combinations)
   - Dhana Yogas (wealth combinations)
   - Parivartana Yoga (sign exchanges)
   - Neecha Bhanga Raja Yoga (debilitation cancellations)
   - Gaja Kesari, Budha-Aditya, and other classical yogas

3. **PLANETARY DIGNITY:**
   - Exaltation/Debilitation status (with exact degrees)
   - Moolatrikona placements
   - Own sign vs enemy sign placements
   - Combustion and retrograde effects

4. **HOUSE ANALYSIS:**
   - Where each house lord is placed
   - What planets aspect each important house
   - Strength of key houses (1st, 9th, 10th, 11th)

5. **LIFE PREDICTIONS (be specific):**
   - Career and professional success patterns
   - Wealth accumulation tendencies
   - Relationships and marriage timing/quality
   - Health vulnerabilities
   - Spiritual growth potential

Format your response using markdown with proper headers (##, ###) and bullet points.
Base all predictions on classical Vedic astrology principles."""
        
        # Format chart data with markdown-friendly formatting
        chart_context = ""
        if chart_data:
            chart_context = "\n\n---\n\n"
            chart_context += "# 🌟 BIRTH CHART ANALYSIS\n\n"
            
            # Ascendant (Lagna)
            if 'ascendant' in chart_data:
                asc = chart_data['ascendant']
                chart_context += "## 📍 Ascendant (Lagna)\n\n"
                chart_context += f"- **Sign:** {asc.get('sign', 'N/A')}\n"
                chart_context += f"- **Degree:** {asc.get('degree_in_sign', 0):.2f}°\n"
                chart_context += f"- **Nakshatra:** {asc.get('nakshatra', 'N/A')} (Pada {asc.get('nakshatra_pada', 'N/A')})\n"
                chart_context += f"- **Nakshatra Lord:** {asc.get('nakshatra_lord', 'N/A')}\n\n"
            
            # Planets with enhanced details
            if 'planets' in chart_data:
                chart_context += "## 🪐 Planetary Positions\n\n"
                
                # Order planets properly
                planet_order = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']
                for planet in planet_order:
                    if planet in chart_data['planets']:
                        pdata = chart_data['planets'][planet]
                        
                        # Build status markers
                        status_markers = []
                        if pdata.get('is_retrograde'):
                            status_markers.append("**RETROGRADE** ℞")
                        if pdata.get('is_combust'):
                            status_markers.append("**COMBUST** ☉")
                        
                        chart_context += f"### {planet}"
                        if status_markers:
                            chart_context += f" [{', '.join(status_markers)}]"
                        chart_context += "\n\n"
                        
                        chart_context += f"- **Sign:** {pdata.get('sign', 'N/A')}\n"
                        chart_context += f"- **House:** {pdata.get('house', 'N/A')}\n"
                        chart_context += f"- **Degree:** {pdata.get('degree_in_sign', 0):.2f}°\n"
                        chart_context += f"- **Nakshatra:** {pdata.get('nakshatra', 'N/A')} (Pada {pdata.get('nakshatra_pada', 'N/A')})\n"
                        chart_context += f"- **Nakshatra Lord:** {pdata.get('nakshatra_lord', 'N/A')}\n\n"
            
            chart_context += "---\n\n"
            chart_context += "## 📋 Analysis Requirements\n\n"
            chart_context += "**IMPORTANT:** You MUST analyze the following in your response:\n\n"
            chart_context += "1. **Planetary Aspects:** Check and describe all major planetary aspects:\n"
            chart_context += "   - Natural aspects (7th house opposition)\n"
            chart_context += "   - Mars aspects: 4th, 7th, 8th houses\n"
            chart_context += "   - Jupiter aspects: 5th, 7th, 9th houses\n"
            chart_context += "   - Saturn aspects: 3rd, 7th, 10th houses\n"
            chart_context += "   - Rahu/Ketu aspects: 5th, 7th, 9th houses\n"
            chart_context += "   - Mutual aspects between planets\n\n"
            chart_context += "2. **Yogas:** Identify beneficial and malefic yogas:\n"
            chart_context += "   - Raja Yogas (power combinations)\n"
            chart_context += "   - Dhana Yogas (wealth combinations)\n"
            chart_context += "   - Parivartana Yoga (sign exchanges)\n"
            chart_context += "   - Neecha Bhanga Raja Yoga (debilitation cancellations)\n"
            chart_context += "   - Other special combinations\n\n"
            chart_context += "3. **Planetary Dignity:** Check exaltation/debilitation status and effects\n\n"
            chart_context += "4. **House Lordships:** Analyze where house lords are placed and their implications\n\n"
            chart_context += "5. **Nakshatra Influence:** Consider nakshatra lords and their impact\n\n"
            chart_context += "6. **Life Predictions:** Provide specific predictions for:\n"
            chart_context += "   - Career and profession\n"
            chart_context += "   - Wealth and finances\n"
            chart_context += "   - Relationships and marriage\n"
            chart_context += "   - Health and longevity\n"
            chart_context += "   - Spiritual inclinations\n\n"
            chart_context += "---\n\n"
        
        # Combine prompt
        full_prompt = system_prompt + chart_context + f"\n\nUser: {user_message}\n\nVedic Astrologer:"
        
        print(f"🔍 Calling {API_PROVIDER.upper()} API ({MODEL_NAME})...")
        print(f"🔍 Prompt length: {len(full_prompt)} chars")
        
        # Call LLM
        assistant_message = call_llm(full_prompt)
        
        print(f"✓ Received response: {len(assistant_message)} chars")
        
        return jsonify({
            'response': assistant_message,
            'model': MODEL_NAME,
            'provider': API_PROVIDER,
            'chart_included': chart_data is not None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'{API_PROVIDER.upper()} API error: {str(e)}',
            'response': f'Error calling {API_PROVIDER} API. Check your API_KEY.'
        }), 500


@app.route('/api/calculate-chart', methods=['POST'])
def calculate_chart():
    """Calculate Vedic astrology chart"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        if lat == 0 and lon == 0:
            return jsonify({'error': 'Valid latitude and longitude required'}), 400
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Store for chat context
        global last_calculated_chart, last_chart_timestamp
        last_calculated_chart = chart
        last_chart_timestamp = dt_local.isoformat()
        
        print(f"✓ Chart calculated and stored")
        
        return jsonify({
            'success': True,
            'chart': chart,
            'timezone_info': {
                'local_time': dt_local.isoformat(),
                'utc_time': dt_utc.isoformat(),
                'offset_hours': timezone_offset
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500


@app.route('/api/execute-tool', methods=['POST'])
def execute_tool():
    """Execute astrological calculation tool"""
    data = request.json
    
    try:
        tool_name = data.get('tool_name')
        arguments = data.get('arguments', {})
        
        if not tool_name:
            return jsonify({'error': 'tool_name required'}), 400
        
        if not last_calculated_chart:
            return jsonify({'error': 'No chart calculated. Calculate chart first.'}), 400
        
        # Build context
        context = {
            "chart": last_calculated_chart,
            "chart_data": last_calculated_chart,
            "jd": 0,
            "birth_time": datetime.utcnow()
        }
        
        # Execute tool
        result = astro_tools.execute_tool(tool_name, arguments, context)
        
        return jsonify({
            'success': True,
            'tool': tool_name,
            'arguments': arguments,
            'result': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Tool execution error: {str(e)}'}), 500


@app.route('/api/generate-rasi-chart', methods=['POST'])
def generate_rasi_chart():
    """Generate Rasi chart visualization data"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Generate Rasi chart structure
        from calculations.core.rasi_chart import generate_rasi_chart_data
        rasi_data = generate_rasi_chart_data(chart)
        
        return jsonify({
            'success': True,
            **rasi_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Rasi chart error: {str(e)}'}), 500


@app.route('/api/generate-divisional-chart', methods=['POST'])
def generate_divisional_chart():
    """Generate divisional chart"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        varga = data.get('varga', 'D9')
        
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate base chart first
        base_chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # For D1 (Rasi), just use the base chart
        if varga == 'D1':
            from calculations.core.rasi_chart import generate_rasi_chart_data
            rasi_data = generate_rasi_chart_data(base_chart)
            return jsonify({
                'success': True,
                **rasi_data
            })
        
        # For other divisional charts, calculate divisional positions
        # Create a new chart structure with divisional positions
        div_chart = {
            'ascendant': {},
            'planets': {}
        }
        
        # Calculate divisional ascendant
        asc_long = base_chart['ascendant']['degree']
        div_asc_long = ephemeris.calculate_divisional_chart(asc_long, varga)
        sign_num = int(div_asc_long / 30)
        div_chart['ascendant'] = {
            'longitude': div_asc_long,
            'sign': ephemeris.SIGNS[sign_num],
            'degree_in_sign': div_asc_long % 30
        }
        
        # Calculate divisional positions for each planet
        for planet_name, planet_data in base_chart['planets'].items():
            planet_long = planet_data['longitude']
            div_planet_long = ephemeris.calculate_divisional_chart(planet_long, varga)
            sign_num = int(div_planet_long / 30)
            
            div_chart['planets'][planet_name] = {
                'longitude': div_planet_long,
                'sign': ephemeris.SIGNS[sign_num],
                'degree_in_sign': div_planet_long % 30,
                'is_retrograde': planet_data.get('is_retrograde', False),
                'is_combust': planet_data.get('is_combust', False)
            }
        
        # Generate chart structure
        from calculations.core.rasi_chart import generate_rasi_chart_data
        rasi_data = generate_rasi_chart_data(div_chart)
        
        return jsonify({
            'success': True,
            **rasi_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Divisional chart error: {str(e)}'}), 500


@app.route('/api/calculate-shadbala', methods=['POST'])
def calculate_shadbala():
    """Calculate Shadbala"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        jd = ephemeris.get_julian_day(dt_utc)
        
        # Calculate Shadbala using class
        from calculations.core.shadbala_enhanced import EnhancedShadbalaCalculator
        calculator = EnhancedShadbalaCalculator()
        
        shadbala_results = {}
        for planet_name, planet_data in chart['planets'].items():
            if planet_name not in ['Rahu', 'Ketu']:
                planet_data_with_name = dict(planet_data)
                planet_data_with_name['name'] = planet_name
                shadbala_results[planet_name] = calculator.calculate_shadbala(
                    planet_data_with_name, chart, jd, dt_utc
                )
        
        return jsonify({
            'success': True,
            'shadbala': shadbala_results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Shadbala error: {str(e)}'}), 500


@app.route('/api/calculate-bhavabala', methods=['POST'])
def calculate_bhavabala():
    """Calculate Bhavabala"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Calculate Bhavabala using class
        from calculations.core.bhavabala_enhanced import EnhancedBhavabalaCalculator
        calculator = EnhancedBhavabalaCalculator()
        bhavabala = calculator.calculate_all_houses(chart)
        
        return jsonify({
            'success': True,
            'bhavabala': bhavabala
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Bhavabala error: {str(e)}'}), 500


@app.route('/api/calculate-dasha', methods=['POST'])
def calculate_dasha():
    """Calculate Vimshottari Dasha"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        moon_longitude = chart['planets']['Moon']['longitude']
        
        # Calculate Dasha using class
        from calculations.core.dasha import VimshottariDasha
        dasha_calc = VimshottariDasha()
        
        current_dasha = dasha_calc.get_current_dasha(dt_utc, moon_longitude)
        timeline = dasha_calc.get_dasha_timeline(dt_utc, moon_longitude, years_ahead=30)
        
        return jsonify({
            'success': True,
            'current_dasha': current_dasha,
            'timeline': timeline
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Dasha error: {str(e)}'}), 500


@app.route('/api/calculate-nakshatra-attributes', methods=['POST'])
def calculate_nakshatra_attributes():
    """Calculate Nakshatra attributes and Karakas"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Calculate chart
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Calculate attributes using class
        from calculations.core.nakshatra_attributes import NakshatraAttributes
        nak_calc = NakshatraAttributes()
        attributes = nak_calc.get_chart_attributes(chart)
        
        return jsonify({
            'success': True,
            'attributes': attributes
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Nakshatra attributes error: {str(e)}'}), 500


@app.route('/api/search-location', methods=['GET'])
def search_location():
    """Search for locations using geocoding"""
    query = request.args.get('q', '')
    
    if not query or len(query) < 3:
        return jsonify({'locations': []})
    
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="vedic-astrology")
        locations = geolocator.geocode(query, exactly_one=False, limit=5)
        
        results = []
        if locations:
            for loc in locations:
                results.append({
                    'name': loc.address,
                    'latitude': loc.latitude,
                    'longitude': loc.longitude
                })
        
        return jsonify({'locations': results})
        
    except Exception as e:
        print(f"Location search error: {e}")
        return jsonify({'locations': []})


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Vedic Astrology Server with Multi-API Support")
    print("=" * 70)
    print(f"Selected Provider: {API_PROVIDER.upper()}")
    print(f"API Status: {'✓ Configured' if llm_client else '✗ Not configured'}")
    if llm_client:
        print(f"Model: {MODEL_NAME}")
    else:
        print(f"\n⚠️  To configure {API_PROVIDER.upper()}:")
        print(f"   1. Copy .env.example to .env")
        print(f"   2. Add your {API_PROVIDER.upper()}_API_KEY in .env")
        print(f"   3. Restart server\n")
    
    print(f"\nCalculation Tools: 9 tools available")
    print(f"Ephemeris: Ready")
    print(f"\nAvailable API Providers:")
    for provider, key in API_KEYS.items():
        status = "✓ Configured" if key else "○ Not configured"
        print(f"  {provider}: {status}")
    
    print("\nStarting server on http://localhost:5001")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)