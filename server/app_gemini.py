"""
Vedic Astrology Server with Gemini API
Uses Google's Gemini for reasoning + local calculations for precision
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from calculations.core.ephemeris import VedicEphemeris
from calculations.core.astro_tools import AstroTools

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Install google-generativeai: pip install google-generativeai")

app = Flask(__name__)
CORS(app)

# Initialize calculators
ephemeris = VedicEphemeris()
astro_tools = AstroTools()

# Store last calculated chart
last_calculated_chart = None
last_chart_timestamp = None

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')  # Set via environment variable
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Free reasoning model
    print("✓ Gemini API configured")
else:
    gemini_model = None
    print("⚠️  Set GEMINI_API_KEY environment variable to use Gemini")


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gemini_available': gemini_model is not None,
        'ephemeris_available': True,
        'tools_available': True
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with Gemini API + local calculation tools
    """
    data = request.json
    user_message = data.get('message', '')
    chart_data = data.get('chart_data', None)
    
    # Use last calculated chart if none provided
    if chart_data is None and last_calculated_chart is not None:
        chart_data = last_calculated_chart
        print(f"✓ Using last calculated chart from {last_chart_timestamp}")
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    if not gemini_model:
        return jsonify({
            'error': 'Gemini API not configured',
            'response': 'Please set GEMINI_API_KEY environment variable to use this feature.'
        }), 503
    
    try:
        # Build system prompt with chart data and tools
        system_prompt = """You are an expert Vedic astrologer with access to precise calculation tools.

You have access to the following tools to get exact astrological calculations:
- calculate_planet_strength: Get Shadbala strength of a planet
- calculate_house_strength: Get Bhavabala strength of a house
- get_navamsa_position: Get D9 Navamsa position
- check_planetary_aspect: Check aspects (includes Rahu/Ketu, mutual aspects)
- check_retrograde_status: Check retrograde status
- check_combustion: Check combustion
- get_current_dasha: Get Vimshottari Dasha
- get_planet_dignity: Get complete dignity (exaltation, debilitation, moolatrikona, own signs)
- check_planetary_friendship: Check planetary friendships

When you need calculations, mention which tool you'd like to use and I'll run it for you.

Analyze the chart data provided and give comprehensive Vedic astrology predictions based on:
1. Planetary positions and their strengths
2. House placements and their significance
3. Aspects between planets
4. Retrograde/combust planets
5. Overall chart patterns"""
        
        # Format chart data
        chart_context = ""
        if chart_data:
            chart_context = "\n\n" + "="*60 + "\n"
            chart_context += "BIRTH CHART ANALYSIS\n"
            chart_context += "="*60 + "\n\n"
            
            # Ascendant
            if 'ascendant' in chart_data:
                asc = chart_data['ascendant']
                chart_context += f"ASCENDANT (Lagna): {asc.get('sign', 'N/A')} at {asc.get('degree_in_sign', 0):.2f}°\n"
                chart_context += f"Nakshatra: {asc.get('nakshatra', 'N/A')} (Pada {asc.get('nakshatra_pada', 'N/A')})\n\n"
            
            # Planets
            if 'planets' in chart_data:
                chart_context += "PLANETARY POSITIONS:\n\n"
                for planet, pdata in chart_data['planets'].items():
                    chart_context += f"{planet}:\n"
                    chart_context += f"  Sign: {pdata.get('sign', 'N/A')}, "
                    chart_context += f"House: {pdata.get('house', 'N/A')}, "
                    chart_context += f"Degree: {pdata.get('degree_in_sign', 0):.2f}°\n"
                    chart_context += f"  Nakshatra: {pdata.get('nakshatra', 'N/A')} (Pada {pdata.get('nakshatra_pada', 'N/A')})\n"
                    if pdata.get('is_retrograde'):
                        chart_context += f"  Status: RETROGRADE ®\n"
                    if pdata.get('is_combust'):
                        chart_context += f"  Status: COMBUST\n"
                    chart_context += "\n"
            
            chart_context += "="*60 + "\n"
        
        # Combine prompt
        full_prompt = system_prompt + chart_context + f"\n\nUser Question: {user_message}"
        
        print(f"🔍 Sending to Gemini API...")
        print(f"🔍 Prompt length: {len(full_prompt)} chars")
        
        # Call Gemini API
        response = gemini_model.generate_content(full_prompt)
        assistant_message = response.text
        
        print(f"✓ Received response: {len(assistant_message)} chars")
        
        # Check if LLM is requesting tool calls
        tool_results = []
        if chart_data and any(tool in assistant_message.lower() for tool in ['calculate_', 'check_', 'get_']):
            print("🔧 LLM mentioned tools - offering to run them")
            
        return jsonify({
            'response': assistant_message,
            'model': 'gemini-2.0-flash-exp',
            'tool_calls': tool_results,
            'gemini_powered': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Gemini API error: {str(e)}',
            'response': 'Error processing request with Gemini API.'
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
        
        # Need chart context
        if not last_calculated_chart:
            return jsonify({'error': 'No chart calculated. Calculate chart first.'}), 400
        
        # Build context
        context = {
            "chart": last_calculated_chart,
            "chart_data": last_calculated_chart,
            "jd": 0,  # Will be calculated from chart if needed
            "birth_time": datetime.utcnow()  # Placeholder
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


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Vedic Astrology Server with Gemini API")
    print("=" * 60)
    print(f"✓ Gemini API: {'Configured' if gemini_model else 'Not configured (set GEMINI_API_KEY)'}")
    print(f"✓ Calculation tools: Ready (9 tools available)")
    print(f"✓ Ephemeris: Ready")
    print("\nStarting server on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)