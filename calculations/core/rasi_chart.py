"""
Rasi Chart Generation for South Indian Style
Generates chart data structure for frontend visualization
"""

def generate_rasi_chart_data(chart_data):
    """
    Generate Rasi chart data in South Indian format
    
    Args:
        chart_data: Chart data from ephemeris with planets and ascendant
        
    Returns:
        dict: Chart data formatted for visualization
    """
    
    # Zodiac signs in order
    signs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    
    # Initialize sign data structure (1-12)
    sign_data = {}
    for i in range(1, 13):
        sign_data[i] = {
            'sign': signs[i-1],
            'planets': [],
            'has_ascendant': False
        }
    
    # Get ascendant sign number (1-12)
    asc_sign = chart_data['ascendant']['sign']
    asc_sign_num = signs.index(asc_sign) + 1
    sign_data[asc_sign_num]['has_ascendant'] = True
    
    # Place planets in their signs
    for planet_name, planet_data in chart_data['planets'].items():
        planet_sign = planet_data['sign']
        planet_sign_num = signs.index(planet_sign) + 1
        
        sign_data[planet_sign_num]['planets'].append({
            'name': planet_name,
            'degree': planet_data['degree_in_sign'],
            'is_retrograde': planet_data.get('is_retrograde', False),
            'is_combust': planet_data.get('is_combust', False)
        })
    
    return {
        'ascendant': {
            'sign': asc_sign,
            'degree': chart_data['ascendant']['degree_in_sign']
        },
        'signs': sign_data
    }