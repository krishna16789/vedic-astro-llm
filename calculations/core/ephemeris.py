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
    
    def calculate_divisional_chart(self, longitude: float, varga: str) -> float:
        """Calculate divisional chart (Varga) position for a planet
        
        Args:
            longitude: Sidereal longitude of planet
            varga: Divisional chart type (D1, D2, D3, D4, D7, D9, D10, D12, D16, D20, D24, D27, D30, D40, D45, D60)
        
        Returns:
            Divisional chart longitude
        """
        # Get varga number from string (e.g., "D9" -> 9)
        varga_num = int(varga[1:])
        
        # Get sign and degree within sign
        sign_num = int(longitude / 30)
        degree_in_sign = longitude % 30
        
        if varga_num == 1:
            # D1 - Rasi (Birth Chart) - no change
            return longitude
        
        elif varga_num == 9:
            # D9 - Navamsa Chart (most important divisional chart)
            # Each sign divided into 9 parts of 3°20' (3.333...)
            # Starting sign depends on nature of birth sign (Parashari system):
            # Movable (0,3,6,9): Aries, Cancer, Libra, Capricorn - Start from same sign
            # Fixed (1,4,7,10): Taurus, Leo, Scorpio, Aquarius - Start from 9th sign (8 signs ahead)
            # Dual (2,5,8,11): Gemini, Virgo, Sagittarius, Pisces - Start from 5th sign (4 signs ahead)
            
            navamsa_within_sign = int(degree_in_sign / (30.0/9.0))  # 0-8
            
            # Determine starting sign based on sign nature
            sign_nature = sign_num % 3
            if sign_nature == 0:  # Movable signs
                start_sign = sign_num
            elif sign_nature == 1:  # Fixed signs
                start_sign = (sign_num + 8) % 12
            else:  # Dual signs (sign_nature == 2)
                start_sign = (sign_num + 4) % 12  # FIXED: was +6, should be +4
            
            # Calculate final navamsa sign
            new_sign = (start_sign + navamsa_within_sign) % 12
            
            # Degree within the navamsa sign
            degree_within_navamsa = (degree_in_sign % (30.0/9.0)) * 9.0
            
            return (new_sign * 30) + degree_within_navamsa
        
        elif varga_num == 2:
            # D2 - Hora Chart (Wealth)
            # Each sign divided into 2 parts of 15° each
            # Odd signs: First hora = Leo, Second hora = Cancer
            # Even signs: First hora = Cancer, Second hora = Leo
            hora_num = int(degree_in_sign / 15.0)  # 0 or 1
            
            if sign_num % 2 == 0:  # Even signs
                new_sign = 4 if hora_num == 0 else 3  # Leo or Cancer
            else:  # Odd signs
                new_sign = 3 if hora_num == 0 else 4  # Cancer or Leo
            
            degree_within_hora = (degree_in_sign % 15.0) * 2.0
            return (new_sign * 30) + degree_within_hora
        
        elif varga_num == 3:
            # D3 - Drekkana Chart (Siblings, courage)
            # Each sign divided into 3 parts of 10° each
            # Start from same sign, then 5th and 9th
            drekkana_num = int(degree_in_sign / 10.0)  # 0, 1, or 2
            new_sign = (sign_num + (drekkana_num * 4)) % 12
            
            degree_within_drekkana = (degree_in_sign % 10.0) * 3.0
            return (new_sign * 30) + degree_within_drekkana
        
        elif varga_num == 4:
            # D4 - Chaturthamsa Chart (Fortune, property)
            # Each sign divided into 4 parts of 7.5° each
            # Start from same sign
            chaturthamsa_num = int(degree_in_sign / 7.5)  # 0, 1, 2, or 3
            new_sign = (sign_num + (chaturthamsa_num * 3)) % 12
            
            degree_within_part = (degree_in_sign % 7.5) * 4.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 7:
            # D7 - Saptamsa Chart (Children, grandchildren)
            # Each sign divided into 7 parts
            # Odd zodiac signs (1,3,5,7,9,11) start from same sign
            # Even zodiac signs (2,4,6,8,10,12) start from 7th sign
            part_num = int(degree_in_sign / (30.0/7.0))  # 0-6
            
            # sign_num is 0-indexed, so odd zodiac signs have even sign_num
            if sign_num % 2 == 0:  # Odd zodiac signs
                start_sign = sign_num
            else:  # Even zodiac signs
                start_sign = (sign_num + 6) % 12
            
            new_sign = (start_sign + part_num) % 12
            degree_within_part = (degree_in_sign % (30.0/7.0)) * 7.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 10:
            # D10 - Dasamsa Chart (Career, profession)
            # Each sign divided into 10 parts of 3° each
            # Odd zodiac signs start from same sign, even zodiac signs start from 9th sign
            part_num = int(degree_in_sign / 3.0)  # 0-9
            
            # sign_num is 0-indexed, so odd zodiac signs have even sign_num
            if sign_num % 2 == 0:  # Odd zodiac signs
                start_sign = sign_num
            else:  # Even zodiac signs
                start_sign = (sign_num + 8) % 12
            
            new_sign = (start_sign + part_num) % 12
            degree_within_part = (degree_in_sign % 3.0) * 10.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 12:
            # D12 - Dwadasamsa Chart (Parents)
            # Each sign divided into 12 parts of 2.5° each
            # Start from same sign
            part_num = int(degree_in_sign / 2.5)  # 0-11
            new_sign = (sign_num + part_num) % 12
            
            degree_within_part = (degree_in_sign % 2.5) * 12.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 16:
            # D16 - Shodasamsa Chart (Vehicles, happiness)
            # Each sign divided into 16 parts of 1.875° each
            # Movable signs start from Aries, Fixed from Leo, Dual from Sagittarius
            part_num = int(degree_in_sign / 1.875)  # 0-15
            
            sign_nature = sign_num % 3
            if sign_nature == 0:  # Movable
                start_sign = 0  # Aries
            elif sign_nature == 1:  # Fixed
                start_sign = 4  # Leo
            else:  # Dual
                start_sign = 8  # Sagittarius
            
            new_sign = (start_sign + part_num) % 12
            degree_within_part = (degree_in_sign % 1.875) * 16.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 20:
            # D20 - Vimsamsa Chart (Spiritual pursuits)
            # Each sign divided into 20 parts of 1.5° each
            # Movable signs start from Aries, Fixed from Sagittarius, Dual from Leo
            part_num = int(degree_in_sign / 1.5)  # 0-19
            
            sign_nature = sign_num % 3
            if sign_nature == 0:  # Movable
                start_sign = 0  # Aries
            elif sign_nature == 1:  # Fixed
                start_sign = 8  # Sagittarius
            else:  # Dual
                start_sign = 4  # Leo
            
            new_sign = (start_sign + part_num) % 12
            degree_within_part = (degree_in_sign % 1.5) * 20.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 24:
            # D24 - Chaturvimsamsa Chart (Education, learning)
            # Each sign divided into 24 parts of 1.25° each
            # Odd zodiac signs start from Leo, even zodiac signs start from Cancer
            part_num = int(degree_in_sign / 1.25)  # 0-23
            
            # sign_num is 0-indexed, so odd zodiac signs have even sign_num
            if sign_num % 2 == 0:  # Odd zodiac signs
                start_sign = 4  # Leo
            else:  # Even zodiac signs
                start_sign = 3  # Cancer
            
            new_sign = (start_sign + part_num) % 12
            degree_within_part = (degree_in_sign % 1.25) * 24.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 27:
            # D27 - Nakshatramsa/Bhamsa Chart (Strengths and weaknesses)
            # Each sign divided into 27 parts (1°06'40" each)
            # The 27 divisions cycle through the 12 signs starting from element-based sign
            part_num = int(degree_in_sign / (30.0/27.0))  # 0-26
            
            # Fire signs start from Aries, Earth from Cancer, Air from Libra, Water from Capricorn
            element = sign_num % 4
            start_signs = [0, 3, 6, 9]  # Aries, Cancer, Libra, Capricorn
            start_sign = start_signs[element]
            
            # Calculate which of the 27 nakshamsa divisions we're in for this sign
            # Then find which zodiac sign that maps to
            total_division = (sign_num * 27) + part_num  # Total division number 0-323
            new_sign = (start_sign + (total_division % 27)) % 12
            
            degree_within_part = (degree_in_sign % (30.0/27.0)) * 27.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 30:
            # D30 - Trimsamsa Chart (Misfortunes, evils)
            # This uses a special non-uniform division
            # Odd zodiac signs: Mars(5°), Saturn(5°), Jupiter(8°), Mercury(7°), Venus(5°)
            # Even zodiac signs: Venus(5°), Mercury(7°), Jupiter(8°), Saturn(5°), Mars(5°)
            
            # sign_num is 0-indexed, so odd zodiac signs have even sign_num
            if sign_num % 2 == 0:  # Odd zodiac signs
                if degree_in_sign < 5:
                    new_sign = 0  # Aries (Mars)
                    degree = degree_in_sign * 6.0
                elif degree_in_sign < 10:
                    new_sign = 9  # Capricorn (Saturn)
                    degree = (degree_in_sign - 5) * 6.0
                elif degree_in_sign < 18:
                    new_sign = 8  # Sagittarius (Jupiter)
                    degree = (degree_in_sign - 10) * 3.75
                elif degree_in_sign < 25:
                    new_sign = 2  # Gemini (Mercury)
                    degree = (degree_in_sign - 18) * 4.2857
                else:
                    new_sign = 1  # Taurus (Venus)
                    degree = (degree_in_sign - 25) * 6.0
            else:  # Even zodiac signs
                if degree_in_sign < 5:
                    new_sign = 1  # Taurus (Venus)
                    degree = degree_in_sign * 6.0
                elif degree_in_sign < 12:
                    new_sign = 2  # Gemini (Mercury)
                    degree = (degree_in_sign - 5) * 4.2857
                elif degree_in_sign < 20:
                    new_sign = 8  # Sagittarius (Jupiter)
                    degree = (degree_in_sign - 12) * 3.75
                elif degree_in_sign < 25:
                    new_sign = 9  # Capricorn (Saturn)
                    degree = (degree_in_sign - 20) * 6.0
                else:
                    new_sign = 0  # Aries (Mars)
                    degree = (degree_in_sign - 25) * 6.0
            
            return (new_sign * 30) + degree
        
        elif varga_num == 40:
            # D40 - Khavedamsa Chart (Auspicious and inauspicious effects)
            # Each sign divided into 40 parts of 0.75° each
            # Traditional method: part cycles through 12 signs
            part_num = int(degree_in_sign / 0.75)  # 0-39
            
            # Use part % 12 to cycle through zodiac starting from current sign
            new_sign = (sign_num + (part_num % 12)) % 12
            degree_within_part = (degree_in_sign % 0.75) * 40.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 45:
            # D45 - Akshavedamsa Chart (General indications)
            # Each sign divided into 45 parts
            # Traditional method: part cycles through 12 signs
            part_num = int(degree_in_sign / (30.0/45.0))  # 0-44
            
            # Use part % 12 to cycle through zodiac starting from current sign
            new_sign = (sign_num + (part_num % 12)) % 12
            degree_within_part = (degree_in_sign % (30.0/45.0)) * 45.0
            return (new_sign * 30) + degree_within_part
        
        elif varga_num == 60:
            # D60 - Shashtiamsa Chart (Overall life, past karma)
            # Each sign divided into 60 parts of 0.5° each
            # Traditional method: part cycles through 12 signs
            part_num = int(degree_in_sign / 0.5)  # 0-59
            
            # Use part % 12 to cycle through zodiac starting from current sign
            new_sign = (sign_num + (part_num % 12)) % 12
            degree_within_part = (degree_in_sign % 0.5) * 60.0
            return (new_sign * 30) + degree_within_part
        
        # For unsupported vargas, return original longitude
        return longitude


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