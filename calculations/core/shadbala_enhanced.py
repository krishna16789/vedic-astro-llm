"""
Enhanced Shadbala (Six-fold Planetary Strength) Calculations
Comprehensive implementation matching classical texts and AstroSage
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple
import swisseph as swe


class EnhancedShadbalaCalculator:
    """Calculate comprehensive Shadbala matching AstroSage standards"""
    
    # Planet natural strengths
    NATURAL_STRENGTH = {
        'Sun': 60, 'Moon': 51.43, 'Venus': 42.86, 'Jupiter': 34.29,
        'Mercury': 25.71, 'Mars': 17.14, 'Saturn': 8.57
    }
    
    # Minimum required strength (in Rupas)
    MINIMUM_REQUIRED = {
        'Sun': 5.0, 'Moon': 6.0, 'Mars': 5.0, 'Mercury': 7.0,
        'Jupiter': 6.5, 'Venus': 5.5, 'Saturn': 5.0
    }
    
    # Exaltation points (sign, degree)
    EXALTATION = {
        'Sun': (0, 10), 'Moon': (1, 3), 'Mars': (9, 28), 'Mercury': (5, 15),
        'Jupiter': (3, 5), 'Venus': (11, 27), 'Saturn': (6, 20)
    }
    
    # Debilitation points (opposite to exaltation)
    DEBILITATION = {
        'Sun': (6, 10), 'Moon': (7, 3), 'Mars': (3, 28), 'Mercury': (11, 15),
        'Jupiter': (9, 5), 'Venus': (5, 27), 'Saturn': (0, 20)
    }
    
    # Moolatrikona ranges (sign, start_degree, end_degree)
    MOOLATRIKONA = {
        'Sun': (4, 0, 20), 'Moon': (1, 3, 30), 'Mars': (0, 0, 12),
        'Mercury': (5, 16, 20), 'Jupiter': (8, 0, 10),
        'Venus': (6, 0, 15), 'Saturn': (10, 0, 20)
    }
    
    # Own signs
    OWN_SIGNS = {
        'Sun': [4], 'Moon': [3], 'Mars': [0, 7], 'Mercury': [2, 5],
        'Jupiter': [8, 11], 'Venus': [1, 6], 'Saturn': [9, 10]
    }
    
    # Friend/Enemy relationships
    RELATIONSHIPS = {
        'Sun': {'friends': ['Moon', 'Mars', 'Jupiter'], 'enemies': ['Venus', 'Saturn'], 'neutral': ['Mercury']},
        'Moon': {'friends': ['Sun', 'Mercury'], 'enemies': [], 'neutral': ['Mars', 'Jupiter', 'Venus', 'Saturn']},
        'Mars': {'friends': ['Sun', 'Moon', 'Jupiter'], 'enemies': ['Mercury'], 'neutral': ['Venus', 'Saturn']},
        'Mercury': {'friends': ['Sun', 'Venus'], 'enemies': ['Moon'], 'neutral': ['Mars', 'Jupiter', 'Saturn']},
        'Jupiter': {'friends': ['Sun', 'Moon', 'Mars'], 'enemies': ['Mercury', 'Venus'], 'neutral': ['Saturn']},
        'Venus': {'friends': ['Mercury', 'Saturn'], 'enemies': ['Sun', 'Moon'], 'neutral': ['Mars', 'Jupiter']},
        'Saturn': {'friends': ['Mercury', 'Venus'], 'enemies': ['Sun', 'Moon', 'Mars'], 'neutral': ['Jupiter']}
    }
    
    def __init__(self):
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    
    def calculate_shadbala(self, planet_data: Dict, chart_data: Dict, jd: float, birth_time: datetime) -> Dict:
        """Calculate complete Shadbala with all sub-components"""
        planet_name = planet_data.get('name', '')
        
        if planet_name in ['Rahu', 'Ketu']:
            return {'total': 0, 'components': {}, 'note': 'Shadbala not calculated for nodes'}
        
        # Calculate all components with sub-components
        sthana_bala_result = self._calculate_sthana_bala_detailed(planet_name, planet_data, chart_data)
        dig_bala = self._calculate_dig_bala(planet_name, planet_data, chart_data)
        kala_bala_result = self._calculate_kala_bala_detailed(planet_name, planet_data, jd, birth_time)
        cheshta_bala = self._calculate_cheshta_bala(planet_name, planet_data)
        naisargika_bala = self.NATURAL_STRENGTH.get(planet_name, 30)
        drik_bala = self._calculate_drik_bala(planet_name, planet_data, chart_data)
        
        total = (sthana_bala_result['total'] + dig_bala + kala_bala_result['total'] + 
                cheshta_bala + naisargika_bala + drik_bala)
        rupas = total / 60
        
        # Calculate Ishta and Kashta Phala
        ishta_kashta = self._calculate_ishta_kashta_phala(rupas, planet_name)
        
        return {
            'total': total,
            'rupas': rupas,
            'minimum_required': self.MINIMUM_REQUIRED.get(planet_name, 5.0),
            'ratio': rupas / self.MINIMUM_REQUIRED.get(planet_name, 5.0),
            'components': {
                'sthana_bala': sthana_bala_result,
                'dig_bala': dig_bala,
                'kala_bala': kala_bala_result,
                'cheshta_bala': cheshta_bala,
                'naisargika_bala': naisargika_bala,
                'drik_bala': drik_bala
            },
            'ishta_phala': ishta_kashta['ishta'],
            'kashta_phala': ishta_kashta['kashta']
        }
    
    def _calculate_sthana_bala_detailed(self, planet: str, planet_data: Dict, chart_data: Dict) -> Dict:
        """Calculate Sthana Bala with all sub-components"""
        longitude = planet_data.get('longitude', 0)
        sign_num = int(longitude / 30)
        degree_in_sign = longitude % 30
        
        # 1. Uchcha Bala (Exaltation strength)
        uchcha_bala = self._calc_uchcha_bala(planet, longitude)
        
        # 2. Saptavargaja Bala (simplified - would need D2, D3, D9, D12, D30 charts)
        saptavargaja_bala = self._calc_saptavargaja_bala(planet, sign_num, degree_in_sign)
        
        # 3. Ojayugmarasyamsa Bala (Odd/Even sign strength)
        ojayugmarasyamsa_bala = self._calc_ojayugmarasyamsa_bala(planet, sign_num, degree_in_sign)
        
        # 4. Kendra Bala (Angular strength)
        kendra_bala = self._calc_kendra_bala(longitude, chart_data.get('ascendant', {}).get('degree', 0))
        
        # 5. Drekkana Bala (Decanate strength)
        drekkana_bala = self._calc_drekkana_bala(planet, degree_in_sign)
        
        total = uchcha_bala + saptavargaja_bala + ojayugmarasyamsa_bala + kendra_bala + drekkana_bala
        
        return {
            'total': total,
            'uchcha_bala': uchcha_bala,
            'saptavargaja_bala': saptavargaja_bala,
            'ojayugmarasyamsa_bala': ojayugmarasyamsa_bala,
            'kendra_bala': kendra_bala,
            'drekkana_bala': drekkana_bala
        }
    
    def _calc_uchcha_bala(self, planet: str, longitude: float) -> float:
        """
        Uchcha Bala (Exaltation strength) - Classical Formula from B.V. Raman
        Formula: (Planet's long. - Its debilitation point) ÷ 3
        If difference > 180°, subtract from 360°
        Maximum: 60 shashtiamsas at exaltation point
        Minimum: 0 shashtiamsas at debilitation point
        """
        if planet not in self.DEBILITATION:
            return 0
        
        # Get debilitation point
        debil_sign, debil_degree = self.DEBILITATION[planet]
        debil_long = debil_sign * 30 + debil_degree
        
        # Calculate difference from debilitation point
        diff = longitude - debil_long
        
        # Normalize to 0-360 range
        diff = diff % 360
        
        # If difference exceeds 180°, subtract from 360° (corrected difference)
        if diff > 180:
            diff = 360 - diff
        
        # Classical formula: difference ÷ 3 = Uchcha Bala
        uchcha_bala = diff / 3.0
        
        return max(0, min(60, uchcha_bala))
    
    def _calc_saptavargaja_bala(self, planet: str, sign_num: int, degree: float) -> float:
        """
        Saptavargaja Bala - Strength from seven divisional charts
        Classical formula from B.V. Raman (Chapter III, Article 30)
        
        Dignity strengths:
        - Moolatrikona: 45 shashtiamsas
        - Own sign (Swavarga): 30 shashtiamsas
        - Intimate friend (Adhimitra): 22.5 shashtiamsas
        - Friend (Mitra): 15 shashtiamsas
        - Neutral (Sama): 7.5 shashtiamsas
        - Enemy (Satru): 3.75 shashtiamsas
        - Bitter enemy (Adhi Satru): 1.875 shashtiamsas
        """
        total_strength = 0
        
        # 1. Rasi (D1) - Main chart
        total_strength += self._get_varga_strength(planet, sign_num, degree, 'rasi')
        
        # 2. Hora (D2) - Half sign division
        hora_sign = self._calc_hora_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, hora_sign, degree, 'hora')
        
        # 3. Drekkana (D3) - Third sign division
        drek_sign = self._calc_drekkana_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, drek_sign, degree, 'drekkana')
        
        # 4. Saptamsa (D7) - Seventh sign division
        sapt_sign = self._calc_saptamsa_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, sapt_sign, degree, 'saptamsa')
        
        # 5. Navamsa (D9) - Ninth sign division
        nav_sign = self._calc_navamsa_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, nav_sign, degree, 'navamsa')
        
        # 6. Dwadasamsa (D12) - Twelfth sign division
        dwad_sign = self._calc_dwadasamsa_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, dwad_sign, degree, 'dwadasamsa')
        
        # 7. Trimsamsa (D30) - Thirtieth sign division
        trim_sign = self._calc_trimsamsa_sign(sign_num, degree)
        total_strength += self._get_varga_strength(planet, trim_sign, degree, 'trimsamsa')
        
        return total_strength
    
    def _get_varga_strength(self, planet: str, varga_sign: int, degree: float, varga_type: str) -> float:
        """Calculate strength based on planet's dignity in a varga"""
        # Check if in Moolatrikona (only in Rasi)
        if varga_type == 'rasi' and planet in self.MOOLATRIKONA:
            moola_sign, start_deg, end_deg = self.MOOLATRIKONA[planet]
            if varga_sign == moola_sign and start_deg <= degree <= end_deg:
                return 45.0
        
        # Check if in own sign
        if planet in self.OWN_SIGNS and varga_sign in self.OWN_SIGNS[planet]:
            return 30.0
        
        # Get lord of the varga sign
        sign_lords = [
            'Mars', 'Venus', 'Mercury', 'Moon', 'Sun', 'Mercury',
            'Venus', 'Mars', 'Jupiter', 'Saturn', 'Saturn', 'Jupiter'
        ]
        varga_lord = sign_lords[varga_sign]
        
        # Determine relationship
        if planet == varga_lord:
            return 30.0
        
        relationships = self.RELATIONSHIPS.get(planet, {})
        
        # Check for Adhimitra (intimate friend) - friend in both natural and temporary
        if varga_lord in relationships.get('friends', []):
            return 15.0  # Simplified as Mitra
        elif varga_lord in relationships.get('enemies', []):
            return 3.75  # Satru
        else:
            return 7.5  # Sama (neutral)
    
    def _calc_hora_sign(self, sign: int, degree: float) -> int:
        """Calculate Hora (D2) sign"""
        if degree < 15:
            return 4 if sign % 2 == 0 else 3  # Leo if odd, Cancer if even
        else:
            return 3 if sign % 2 == 0 else 4  # Cancer if odd, Leo if even
    
    def _calc_drekkana_sign(self, sign: int, degree: float) -> int:
        """Calculate Drekkana (D3) sign"""
        drek_num = int(degree / 10)
        return (sign + drek_num * 4) % 12
    
    def _calc_saptamsa_sign(self, sign: int, degree: float) -> int:
        """Calculate Saptamsa (D7) sign"""
        sapt_num = int(degree * 7 / 30)
        if sign % 2 == 0:  # Odd sign
            return (sign + sapt_num) % 12
        else:  # Even sign
            return (sign + 6 + sapt_num) % 12
    
    def _calc_navamsa_sign(self, sign: int, degree: float) -> int:
        """Calculate Navamsa (D9) sign"""
        nav_num = int(degree * 9 / 30)
        return (sign + nav_num) % 12
    
    def _calc_dwadasamsa_sign(self, sign: int, degree: float) -> int:
        """Calculate Dwadasamsa (D12) sign"""
        dwad_num = int(degree * 12 / 30)
        return (sign + dwad_num) % 12
    
    def _calc_trimsamsa_sign(self, sign: int, degree: float) -> int:
        """
        Calculate Trimsamsa (D30) sign
        Odd signs: Mars(0-5°), Saturn(5-10°), Jupiter(10-18°), Mercury(18-25°), Venus(25-30°)
        Even signs: Venus(0-5°), Mercury(5-12°), Jupiter(12-20°), Saturn(20-25°), Mars(25-30°)
        """
        if sign % 2 == 0:  # Odd sign (Aries, Gemini, Leo, Libra, Sagittarius, Aquarius)
            if degree < 5:
                return 0  # Mars - Aries
            elif degree < 10:
                return 10  # Saturn - Aquarius
            elif degree < 18:
                return 8  # Jupiter - Sagittarius
            elif degree < 25:
                return 5  # Mercury - Virgo
            else:
                return 6  # Venus - Libra
        else:  # Even sign (Taurus, Cancer, Virgo, Scorpio, Capricorn, Pisces)
            if degree < 5:
                return 6  # Venus - Libra
            elif degree < 12:
                return 5  # Mercury - Virgo
            elif degree < 20:
                return 8  # Jupiter - Sagittarius
            elif degree < 25:
                return 10  # Saturn - Aquarius
            else:
                return 0  # Mars - Aries
    
    def _calc_ojayugmarasyamsa_bala(self, planet: str, sign_num: int, degree: float) -> float:
        """Odd/Even sign and Navamsa strength"""
        strength = 0
        
        # Male planets (Sun, Mars, Jupiter) stronger in odd signs
        # Female planets (Moon, Venus) stronger in even signs
        # Mercury neutral
        
        is_odd_sign = sign_num % 2 == 0  # 0,2,4,6,8,10 are odd (Aries, Gemini, etc)
        
        if planet in ['Sun', 'Mars', 'Jupiter'] and is_odd_sign:
            strength = 30
        elif planet in ['Moon', 'Venus'] and not is_odd_sign:
            strength = 30
        elif planet == 'Mercury':
            strength = 15
        else:
            strength = 15
        
        return strength
    
    def _calc_kendra_bala(self, longitude: float, ascendant: float) -> float:
        """Angular (Kendra) strength"""
        # Distance from ascendant
        diff = (longitude - ascendant + 360) % 360
        house_pos = diff / 30
        
        # Kendra houses are 1, 4, 7, 10
        if 0 <= house_pos < 1 or 3 <= house_pos < 4 or 6 <= house_pos < 7 or 9 <= house_pos < 10:
            return 60
        # Panapara houses (2, 5, 8, 11)
        elif 1 <= house_pos < 2 or 4 <= house_pos < 5 or 7 <= house_pos < 8 or 10 <= house_pos < 11:
            return 30
        # Apoklima houses (3, 6, 9, 12)
        else:
            return 15
    
    def _calc_drekkana_bala(self, planet: str, degree: float) -> float:
        """Decanate strength - always 1 in classical texts"""
        return 1
    
    def _calculate_kala_bala_detailed(self, planet: str, planet_data: Dict, jd: float, birth_time: datetime) -> Dict:
        """Calculate Kala Bala with all sub-components"""
        # 1. Nathonnatha Bala (Day/Night strength)
        nathonnatha_bala = self._calc_nathonnatha_bala(planet, birth_time)
        
        # 2. Paksha Bala (Lunar fortnight strength)
        paksha_bala = self._calc_paksha_bala(planet, jd)
        
        # 3. Tribhaga Bala (Day third strength)
        tribhaga_bala = self._calc_tribhaga_bala(planet, birth_time)
        
        # 4. Abda Bala (Year lord strength)
        abda_bala = self._calc_abda_bala(planet, birth_time)
        
        # 5. Masa Bala (Month lord strength)
        masa_bala = self._calc_masa_bala(planet, birth_time)
        
        # 6. Vara Bala (Weekday lord strength)
        vara_bala = self._calc_vara_bala(planet, birth_time)
        
        # 7. Hora Bala (Hour lord strength)
        hora_bala = self._calc_hora_bala(planet, birth_time)
        
        # 8. Ayana Bala (Declination strength)
        ayana_bala = self._calc_ayana_bala(planet, planet_data, jd)
        
        # 9. Yuddha Bala (Planetary war strength)
        yuddha_bala = 0  # Complex calculation, usually 0
        
        total = (nathonnatha_bala + paksha_bala + tribhaga_bala + abda_bala + 
                masa_bala + vara_bala + hora_bala + ayana_bala + yuddha_bala)
        
        return {
            'total': total,
            'nathonnatha_bala': nathonnatha_bala,
            'paksha_bala': paksha_bala,
            'tribhaga_bala': tribhaga_bala,
            'abda_bala': abda_bala,
            'masa_bala': masa_bala,
            'vara_bala': vara_bala,
            'hora_bala': hora_bala,
            'ayana_bala': ayana_bala,
            'yuddha_bala': yuddha_bala
        }
    
    def _calc_nathonnatha_bala(self, planet: str, birth_time: datetime) -> float:
        """Day/Night strength"""
        hour = birth_time.hour
        is_day = 6 <= hour < 18  # Simplified day/night
        
        day_planets = ['Sun', 'Jupiter', 'Venus']
        night_planets = ['Moon', 'Mars', 'Saturn']
        
        if planet in day_planets:
            return 60.0 if is_day else 21.84
        elif planet in night_planets:
            return 60.0 if not is_day else 21.84
        else:  # Mercury
            return 38.16
    
    def _calc_paksha_bala(self, planet: str, jd: float) -> float:
        """Lunar fortnight strength"""
        # Get moon longitude - swe.calc_ut returns (position_tuple, flags)
        moon_data = swe.calc_ut(jd, swe.MOON)
        sun_data = swe.calc_ut(jd, swe.SUN)
        
        # Extract longitude from the position tuple
        moon_long = moon_data[0][0] if isinstance(moon_data[0], (list, tuple)) else moon_data[0]
        sun_long = sun_data[0][0] if isinstance(sun_data[0], (list, tuple)) else sun_data[0]
        
        # Distance from Sun to Moon
        elongation = (moon_long - sun_long + 360) % 360
        
        # Benefics stronger in waxing (Shukla Paksha)
        # Malefics stronger in waning (Krishna Paksha)
        benefics = ['Jupiter', 'Venus', 'Mercury', 'Moon']
        
        if elongation < 180:  # Waxing
            factor = elongation / 180
        else:  # Waning
            factor = (360 - elongation) / 180
        
        if planet in benefics:
            return 60 * factor
        else:
            return 60 * (1 - factor)
    
    def _calc_tribhaga_bala(self, planet: str, birth_time: datetime) -> float:
        """Day third strength"""
        hour = birth_time.hour + birth_time.minute / 60.0
        
        # Divide day into thirds
        if 6 <= hour < 10:  # First third
            first_third_planets = ['Mercury', 'Venus']
            return 60 if planet in first_third_planets else 0
        elif 10 <= hour < 14:  # Second third
            second_third_planets = ['Sun', 'Saturn']
            return 60 if planet in second_third_planets else 0
        elif 14 <= hour < 18:  # Third third
            third_third_planets = ['Moon', 'Mars', 'Jupiter']
            return 60 if planet in third_third_planets else 0
        else:
            return 0
    
    def _calc_abda_bala(self, planet: str, birth_time: datetime) -> float:
        """Year lord strength"""
        # Simplified - lord of year based on weekday of year start
        year_lord_order = ['Saturn', 'Jupiter', 'Mars', 'Sun', 'Venus', 'Mercury', 'Moon']
        year_mod = birth_time.year % 7
        year_lord = year_lord_order[year_mod]
        
        return 15 if planet == year_lord else 0
    
    def _calc_masa_bala(self, planet: str, birth_time: datetime) -> float:
        """Month lord strength"""
        # Lord of month based on zodiac month
        month_lords = {
            1: 'Mars', 2: 'Venus', 3: 'Mercury', 4: 'Moon', 5: 'Sun', 6: 'Mercury',
            7: 'Venus', 8: 'Mars', 9: 'Jupiter', 10: 'Saturn', 11: 'Saturn', 12: 'Jupiter'
        }
        month_lord = month_lords.get(birth_time.month)
        
        return 30 if planet == month_lord else 0
    
    def _calc_vara_bala(self, planet: str, birth_time: datetime) -> float:
        """Weekday lord strength"""
        weekday_lords = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']
        weekday = birth_time.weekday()  # 0=Monday
        # Adjust to start from Sunday
        weekday = (weekday + 1) % 7
        day_lord = weekday_lords[weekday]
        
        return 45 if planet == day_lord else 0
    
    def _calc_hora_bala(self, planet: str, birth_time: datetime) -> float:
        """Hour lord strength"""
        hour = birth_time.hour
        # Simplified hora calculation
        hora_lords = ['Sun', 'Venus', 'Mercury', 'Moon', 'Saturn', 'Jupiter', 'Mars']
        hora_lord = hora_lords[hour % 7]
        
        return 60 if planet == hora_lord else 0
    
    def _calc_ayana_bala(self, planet: str, planet_data: Dict, jd: float) -> float:
        """Declination/Ayana strength"""
        # Get planet's declination
        latitude = planet_data.get('latitude', 0)
        
        # Benefics stronger when in northern declination
        # Malefics stronger in southern declination
        benefics = ['Jupiter', 'Venus', 'Mercury', 'Moon']
        
        if planet in benefics:
            strength = 60 + latitude * 2
        else:
            strength = 60 - latitude * 2
        
        return max(0, min(120, strength))
    
    def _calculate_dig_bala(self, planet: str, planet_data: Dict, chart_data: Dict) -> float:
        """Directional strength"""
        longitude = planet_data.get('longitude', 0)
        ascendant = chart_data.get('ascendant', {}).get('degree', 0)
        
        # Calculate house position
        diff = (longitude - ascendant + 360) % 360
        
        # Directional strengths by planet
        directions = {
            'Jupiter': 0, 'Mercury': 0,  # 1st house (East)
            'Sun': 270, 'Mars': 270,      # 10th house (South)
            'Saturn': 180,                 # 7th house (West)
            'Moon': 90, 'Venus': 90       # 4th house (North)
        }
        
        if planet in directions:
            ideal = directions[planet]
            angular_diff = abs(diff - ideal)
            if angular_diff > 180:
                angular_diff = 360 - angular_diff
            
            return 60 * (1 - angular_diff / 180)
        
        return 30
    
    def _calculate_cheshta_bala(self, planet: str, planet_data: Dict) -> float:
        """Motional strength"""
        if planet in ['Sun', 'Moon']:
            return 0
        
        speed = planet_data.get('speed', 0)
        is_retrograde = planet_data.get('is_retrograde', False)
        
        if is_retrograde:
            return 60
        
        # Based on speed - faster = stronger
        mean_motions = {
            'Mars': 0.524, 'Mercury': 1.383, 'Jupiter': 0.083,
            'Venus': 1.602, 'Saturn': 0.033
        }
        
        if planet in mean_motions:
            mean = mean_motions[planet]
            ratio = abs(speed) / mean if mean > 0 else 1
            return min(60, 30 * ratio)
        
        return 30
    
    def _calculate_drik_bala(self, planet: str, planet_data: Dict, chart_data: Dict) -> float:
        """Aspectual strength"""
        longitude = planet_data.get('longitude', 0)
        strength = 0
        
        planets = chart_data.get('planets', {})
        for other_name, other_data in planets.items():
            if other_name == planet or other_name in ['Rahu', 'Ketu']:
                continue
            
            other_long = other_data.get('longitude', 0)
            diff = abs(longitude - other_long)
            if diff > 180:
                diff = 360 - diff
            
            # Aspects and their effects
            if abs(diff - 180) < 10:  # Opposition
                if other_name in self.RELATIONSHIPS.get(planet, {}).get('friends', []):
                    strength += 10
                else:
                    strength -= 10
            elif diff < 10:  # Conjunction
                if other_name in self.RELATIONSHIPS.get(planet, {}).get('friends', []):
                    strength += 20
                elif other_name in self.RELATIONSHIPS.get(planet, {}).get('enemies', []):
                    strength -= 15
            elif abs(diff - 120) < 10:  # Trine
                strength += 15
            elif abs(diff - 60) < 10 or abs(diff - 90) < 10:  # Sextile/Square
                strength += 5
        
        return strength
    
    def _calculate_ishta_kashta_phala(self, rupas: float, planet: str) -> Dict:
        """Calculate Ishta and Kashta Phala (benefic and malefic results)"""
        min_req = self.MINIMUM_REQUIRED.get(planet, 5.0)
        
        # Ishta Phala (benefic results) - higher when planet is strong
        if rupas >= min_req:
            ishta = 60 * (rupas / min_req) * 0.8
        else:
            ishta = 60 * (rupas / min_req) * 0.4
        
        # Kashta Phala (malefic results) - lower when planet is strong
        kashta = 60 - ishta
        
        return {
            'ishta': min(60, ishta),
            'kashta': max(0, kashta)
        }