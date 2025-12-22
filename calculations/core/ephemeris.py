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
    is_retrograde: bool = False
    is_combust: bool = False
    combustion_distance: float = 0.0


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
    
    def get_planet_position(self, planet_name: str, jd: float, sun_longitude: float = None) -> PlanetPosition:
        """Get planet position at Julian Day
        
        Args:
            planet_name: Name of the planet
            jd: Julian Day
            sun_longitude: Sun's longitude (for combustion calculation)
        """
        if planet_name not in self.PLANETS:
            raise ValueError(f"Unknown planet: {planet_name}")
        
        planet_id = self.PLANETS[planet_name]
        
        # Calculate position (sidereal) with speed
        # FLG_SIDEREAL for sidereal zodiac, FLG_SPEED for velocity calculation
        result = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL | swe.FLG_SPEED)
        
        longitude = result[0][0]  # Nirayana longitude
        latitude = result[0][1]
        distance = result[0][2]
        speed = result[0][3]  # Speed in longitude (degrees per day)
        
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
        
        # Detect retrograde motion
        # Rahu and Ketu are ALWAYS retrograde by their very nature
        # For other planets, check if speed is negative
        if planet_name in ['Rahu', 'Ketu']:
            is_retrograde = True
        else:
            is_retrograde = speed < 0
        
        # Calculate combustion if Sun longitude is provided
        is_combust = False
        combustion_distance = 0.0
        if sun_longitude is not None and planet_name not in ['Sun', 'Rahu', 'Ketu']:
            combustion_distance = self._calculate_angular_distance(longitude, sun_longitude)
            is_combust = self._is_combust(planet_name, combustion_distance)
        
        return PlanetPosition(
            planet=planet_name,
            longitude=longitude,
            latitude=latitude,
            distance=distance,
            speed=speed,
            sign=sign,
            degree_in_sign=degree_in_sign,
            nakshatra=nakshatra,
            nakshatra_pada=pada,
            is_retrograde=is_retrograde,
            is_combust=is_combust,
            combustion_distance=combustion_distance
        )
    
    def _calculate_angular_distance(self, lon1: float, lon2: float) -> float:
        """Calculate shortest angular distance between two longitudes"""
        diff = abs(lon1 - lon2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _is_combust(self, planet_name: str, distance: float) -> bool:
        """Check if planet is combust (too close to Sun)
        
        Combustion distances (degrees from Sun):
        - Moon: 12°
        - Mars: 17°
        - Mercury: 14° (12° when retrograde)
        - Jupiter: 11°
        - Venus: 10° (8° when retrograde)
        - Saturn: 15°
        """
        combustion_orbs = {
            'Moon': 12.0,
            'Mars': 17.0,
            'Mercury': 14.0,
            'Jupiter': 11.0,
            'Venus': 10.0,
            'Saturn': 15.0
        }
        
        orb = combustion_orbs.get(planet_name, 0)
        return distance <= orb if orb > 0 else False
    
    def get_all_planets(self, dt: datetime) -> Dict[str, PlanetPosition]:
        """Get positions of all planets with retrograde and combustion detection"""
        jd = self.get_julian_day(dt)
        
        # First, get Sun's position for combustion calculations
        sun_pos = self.get_planet_position('Sun', jd)
        sun_longitude = sun_pos.longitude
        
        # Get all planet positions with combustion check
        positions = {}
        for planet_name in self.PLANETS.keys():
            positions[planet_name] = self.get_planet_position(planet_name, jd, sun_longitude)
        
        return positions
    
    def get_ascendant(self, dt: datetime, lat: float, lon: float) -> float:
        """Calculate Ascendant (Lagna) in sidereal zodiac
        
        IMPORTANT: dt must be in UTC time for Swiss Ephemeris calculations
        
        When using FLG_SIDEREAL flag:
        - ascmc[0] gives the actual ascendant degree (sidereal)
        - cusps[1] gives House 1 cusp which may differ from ascendant
        """
        jd = self.get_julian_day(dt)
        
        # Set sidereal mode before calling houses
        swe.set_sid_mode(self.AYANAMSA)
        
        # Call houses_ex with SIDEREAL flag
        cusps, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
        
        # ascmc[0] is the ascendant (already in sidereal zodiac)
        # Note: cusps[1] is House 1 cusp which can differ from ascendant
        ascendant_sidereal = ascmc[0]
        
        return ascendant_sidereal
    
    def get_house_cusps(self, dt: datetime, lat: float, lon: float) -> List[float]:
        """Get all 12 house cusps in sidereal zodiac using Placidus system
        
        IMPORTANT: dt must be in UTC time for Swiss Ephemeris calculations
        
        Returns a list of 12 house cusp longitudes (houses 1-12)
        """
        jd = self.get_julian_day(dt)
        
        # Set sidereal mode before calling houses
        swe.set_sid_mode(self.AYANAMSA)
        
        # Get sidereal houses using Placidus system
        # cusps is a tuple with indices 0-12 (13 elements)
        # Index 0 is unused, indices 1-12 are the 12 house cusps
        cusps, ascmc = swe.houses_ex(jd, lat, lon, b'P', swe.FLG_SIDEREAL)
        
        # Convert tuple to list and extract houses 1-12
        # Swiss Ephemeris returns cusps[0] as unused, cusps[1-12] as the 12 houses
        house_list = list(cusps)
        
        # Return only the 12 house cusps (skip index 0)
        if len(house_list) >= 13:
            return house_list[1:13]
        else:
            # Fallback: take all elements from index 1 onwards and pad if needed
            result = house_list[1:]
            while len(result) < 12:
                result.append(0.0)  # Pad with zeros if somehow we have fewer cusps
            return result[:12]  # Ensure exactly 12 elements
    
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
                    'speed': pos.speed,
                    'is_retrograde': pos.is_retrograde,
                    'is_combust': pos.is_combust,
                    'combustion_distance': pos.combustion_distance
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