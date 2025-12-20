"""
Vedic Astrology Astronomical Calculations using Swiss Ephemeris
Python wrapper for core VedAstro calculation logic
"""

import swisseph as swe
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PlanetPosition:
    """Planetary position data"""
    planet: str
    longitude: float  # Nirayana (sidereal) longitude in degrees
    latitude: float
    distance: float
    speed: float
    sign: str
    degree_in_sign: float
    nakshatra: str
    nakshatra_pada: int


class VedicEphemeris:
    """
    Vedic astrology calculations using Swiss Ephemeris
    Based on VedAstro's C# implementation
    """
    
    # Ayanamsa (precession correction)
    AYANAMSA = swe.SIDM_LAHIRI  # Lahiri ayanamsa (most common)
    
    # Planet IDs
    PLANETS = {
        'Sun': swe.SUN,
        'Moon': swe.MOON,
        'Mars': swe.MARS,
        'Mercury': swe.MERCURY,
        'Jupiter': swe.JUPITER,
        'Venus': swe.VENUS,
        'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE,  # North Node
        'Ketu': swe.MEAN_NODE,  # South Node (180° from Rahu)
    }
    
    # Zodiac signs
    SIGNS = [
        'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
    ]
    
    # Nakshatras (27 lunar mansions)
    NAKSHATRAS = [
        'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra',
        'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'Purva Phalguni', 'Uttara Phalguni',
        'Hasta', 'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha',
        'Mula', 'Purva Ashadha', 'Uttara Ashadha', 'Shravana', 'Dhanishta', 'Shatabhisha',
        'Purva Bhadrapada', 'Uttara Bhadrapada', 'Revati'
    ]
    
    def __init__(self):
        """Initialize Swiss Ephemeris"""
        # Set ayanamsa
        swe.set_sid_mode(self.AYANAMSA)
        
    def get_julian_day(self, dt: datetime) -> float:
        """Convert datetime to Julian Day"""
        return swe.julday(
            dt.year, dt.month, dt.day,
            dt.hour + dt.minute/60.0 + dt.second/3600.0
        )
    
    def get_planet_position(self, planet_name: str, jd: float) -> PlanetPosition:
        """Get planet position at Julian Day"""
        if planet_name not in self.PLANETS:
            raise ValueError(f"Unknown planet: {planet_name}")
        
        planet_id = self.PLANETS[planet_name]
        
        # Calculate position (sidereal)
        result = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
        
        longitude = result[0][0]  # Nirayana longitude
        latitude = result[0][1]
        distance = result[0][2]
        speed = result[0][3]
        
        # Handle Ketu (180° from Rahu)
        if planet_name == 'Ketu':
            longitude = (longitude + 180) % 360
        
        # Calculate sign
        sign_num = int(longitude / 30)
        sign = self.SIGNS[sign_num]
        degree_in_sign = longitude % 30
        
        # Calculate nakshatra
        nakshatra_num = int(longitude / 13.333333333)
        nakshatra = self.NAKSHATRAS[nakshatra_num % 27]
        
        # Calculate pada (quarter of nakshatra)
        pada = int((longitude % 13.333333333) / 3.333333333) + 1
        
        return PlanetPosition(
            planet=planet_name,
            longitude=longitude,
            latitude=latitude,
            distance=distance,
            speed=speed,
            sign=sign,
            degree_in_sign=degree_in_sign,
            nakshatra=nakshatra,
            nakshatra_pada=pada
        )
    
    def get_all_planets(self, dt: datetime) -> Dict[str, PlanetPosition]:
        """Get positions of all planets"""
        jd = self.get_julian_day(dt)
        
        positions = {}
        for planet_name in self.PLANETS.keys():
            positions[planet_name] = self.get_planet_position(planet_name, jd)
        
        return positions
    
    def get_ascendant(self, dt: datetime, lat: float, lon: float) -> float:
        """Calculate ascendant (Lagna)"""
        jd = self.get_julian_day(dt)
        
        # Calculate houses
        houses = swe.houses_ex(jd, lat, lon, b'P')  # Placidus system
        ascendant = houses[1][0]  # First house cusp (sidereal)
        
        return ascendant
    
    def get_house_cusps(self, dt: datetime, lat: float, lon: float) -> List[float]:
        """Get all 12 house cusps"""
        jd = self.get_julian_day(dt)
        
        houses = swe.houses_ex(jd, lat, lon, b'P')
        cusps = houses[0][:12]  # 12 house cusps
        
        return cusps
    
    def get_chart_data(
        self, 
        dt: datetime, 
        lat: float, 
        lon: float
    ) -> Dict:
        """Get complete chart data"""
        # Get planetary positions
        planets = self.get_all_planets(dt)
        
        # Get ascendant
        ascendant = self.get_ascendant(dt, lat, lon)
        ascendant_sign_num = int(ascendant / 30)
        ascendant_sign = self.SIGNS[ascendant_sign_num]
        
        # Get house cusps
        house_cusps = self.get_house_cusps(dt, lat, lon)
        
        return {
            'datetime': dt.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'ascendant': {
                'degree': ascendant,
                'sign': ascendant_sign,
                'degree_in_sign': ascendant % 30
            },
            'planets': {
                name: {
                    'longitude': pos.longitude,
                    'sign': pos.sign,
                    'degree_in_sign': pos.degree_in_sign,
                    'nakshatra': pos.nakshatra,
                    'pada': pos.nakshatra_pada,
                    'speed': pos.speed
                }
                for name, pos in planets.items()
            },
            'house_cusps': house_cusps
        }


# Example usage
if __name__ == "__main__":
    eph = VedicEphemeris()
    
    # Example: Get chart for a specific date, time, and location
    birth_time = datetime(1990, 1, 1, 10, 30, 0)
    latitude = 28.6139  # New Delhi
    longitude = 77.2090
    
    chart = eph.get_chart_data(birth_time, latitude, longitude)
    
    print("Vedic Astrology Chart")
    print("=" * 60)
    print(f"Date/Time: {chart['datetime']}")
    print(f"Location: {chart['latitude']}, {chart['longitude']}")
    print(f"\nAscendant: {chart['ascendant']['sign']} {chart['ascendant']['degree_in_sign']:.2f}°")
    print("\nPlanetary Positions:")
    for planet, data in chart['planets'].items():
        print(f"  {planet:10s}: {data['sign']:12s} {data['degree_in_sign']:6.2f}° "
              f"({data['nakshatra']} - {data['pada']})")