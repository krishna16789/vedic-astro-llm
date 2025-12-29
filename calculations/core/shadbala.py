"""
Shadbala (Six-fold Planetary Strength) Calculations
Based on Parasara's classical Vedic Astrology principles
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple
import swisseph as swe


class ShadbalaCalculator:
    """Calculate Shadbala (Six-fold strength) for planets"""
    
    # Planet natural strengths (in descending order)
    NATURAL_STRENGTH = {
        'Sun': 60,
        'Moon': 51.43,
        'Venus': 42.86,
        'Jupiter': 34.29,
        'Mercury': 25.71,
        'Mars': 17.14,
        'Saturn': 8.57
    }
    
    # Friend/Enemy/Neutral relationships
    RELATIONSHIPS = {
        'Sun': {'friends': ['Moon', 'Mars', 'Jupiter'], 'enemies': ['Venus', 'Saturn'], 'neutral': ['Mercury']},
        'Moon': {'friends': ['Sun', 'Mercury'], 'enemies': [], 'neutral': ['Mars', 'Jupiter', 'Venus', 'Saturn']},
        'Mars': {'friends': ['Sun', 'Moon', 'Jupiter'], 'enemies': ['Mercury'], 'neutral': ['Venus', 'Saturn']},
        'Mercury': {'friends': ['Sun', 'Venus'], 'enemies': ['Moon'], 'neutral': ['Mars', 'Jupiter', 'Saturn']},
        'Jupiter': {'friends': ['Sun', 'Moon', 'Mars'], 'enemies': ['Mercury', 'Venus'], 'neutral': ['Saturn']},
        'Venus': {'friends': ['Mercury', 'Saturn'], 'enemies': ['Sun', 'Moon'], 'neutral': ['Mars', 'Jupiter']},
        'Saturn': {'friends': ['Mercury', 'Venus'], 'enemies': ['Sun', 'Moon', 'Mars'], 'neutral': ['Jupiter']}
    }
    
    # Exaltation degrees (sign, degree)
    EXALTATION = {
        'Sun': (0, 10),      # Aries 10°
        'Moon': (1, 3),      # Taurus 3°
        'Mars': (9, 28),     # Capricorn 28°
        'Mercury': (5, 15),  # Virgo 15°
        'Jupiter': (3, 5),   # Cancer 5°
        'Venus': (11, 27),   # Pisces 27°
        'Saturn': (6, 20)    # Libra 20°
    }
    
    # Moolatrikona ranges (sign, start_degree, end_degree)
    MOOLATRIKONA = {
        'Sun': (4, 0, 20),      # Leo 0-20°
        'Moon': (1, 3, 30),     # Taurus 3-30°
        'Mars': (0, 0, 12),     # Aries 0-12°
        'Mercury': (5, 15, 20), # Virgo 15-20°
        'Jupiter': (8, 0, 10),  # Sagittarius 0-10°
        'Venus': (6, 0, 15),    # Libra 0-15°
        'Saturn': (10, 0, 20)   # Aquarius 0-20°
    }
    
    # Own signs
    OWN_SIGNS = {
        'Sun': [4],           # Leo
        'Moon': [3],          # Cancer
        'Mars': [0, 7],       # Aries, Scorpio
        'Mercury': [2, 5],    # Gemini, Virgo
        'Jupiter': [8, 11],   # Sagittarius, Pisces
        'Venus': [1, 6],      # Taurus, Libra
        'Saturn': [9, 10]     # Capricorn, Aquarius
    }
    
    def __init__(self):
        """Initialize Shadbala calculator"""
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    
    def calculate_shadbala(self, planet_data: Dict, chart_data: Dict, jd: float) -> Dict:
        """Calculate complete Shadbala for a planet
        
        Args:
            planet_data: Planet position data (longitude, latitude, etc.)
            chart_data: Complete chart data with all planets
            jd: Julian Day
            
        Returns:
            Dictionary with all Shadbala components
        """
        planet_name = planet_data.get('name', '')
        
        if planet_name in ['Rahu', 'Ketu']:
            return {'total': 0, 'components': {}, 'note': 'Shadbala not calculated for nodes'}
        
        # Calculate six components
        sthana_bala = self._calculate_sthana_bala(planet_name, planet_data, chart_data)
        dig_bala = self._calculate_dig_bala(planet_name, planet_data, chart_data)
        kala_bala = self._calculate_kala_bala(planet_name, planet_data, jd)
        cheshta_bala = self._calculate_cheshta_bala(planet_name, planet_data)
        naisargika_bala = self._calculate_naisargika_bala(planet_name)
        drik_bala = self._calculate_drik_bala(planet_name, planet_data, chart_data)
        
        total = sthana_bala + dig_bala + kala_bala + cheshta_bala + naisargika_bala + drik_bala
        
        return {
            'total': total,
            'rupas': total / 60,  # Convert to Rupas (1 Rupa = 60 Shashtiamsas)
            'components': {
                'sthana_bala': sthana_bala,
                'dig_bala': dig_bala,
                'kala_bala': kala_bala,
                'cheshta_bala': cheshta_bala,
                'naisargika_bala': naisargika_bala,
                'drik_bala': drik_bala
            }
        }
    
    def _calculate_sthana_bala(self, planet: str, planet_data: Dict, chart_data: Dict) -> float:
        """Calculate Sthana Bala (Positional Strength)"""
        longitude = planet_data.get('longitude', 0)
        sign_num = int(longitude / 30)
        degree_in_sign = longitude % 30
        
        strength = 0
        
        # 1. Uchcha Bala (Exaltation strength)
        if planet in self.EXALTATION:
            exalt_sign, exalt_degree = self.EXALTATION[planet]
            exalt_long = exalt_sign * 30 + exalt_degree
            debil_long = (exalt_long + 180) % 360
            
            diff = abs(longitude - exalt_long)
            if diff > 180:
                diff = 360 - diff
            
            uchcha_bala = 60 * (1 - diff / 180)
            strength += uchcha_bala
        
        # 2. Saptavargaja Bala (Seven divisional chart strengths)
        # Simplified version - in reality calculates D1, D2, D3, D9, D12, D30
        if planet in self.MOOLATRIKONA:
            mt_sign, mt_start, mt_end = self.MOOLATRIKONA[planet]
            if sign_num == mt_sign and mt_start <= degree_in_sign <= mt_end:
                strength += 45
        
        if planet in self.OWN_SIGNS and sign_num in self.OWN_SIGNS[planet]:
            strength += 30
        
        return strength
    
    def _calculate_dig_bala(self, planet: str, planet_data: Dict, chart_data: Dict) -> float:
        """Calculate Dig Bala (Directional Strength)"""
        longitude = planet_data.get('longitude', 0)
        ascendant = chart_data.get('ascendant', {}).get('degree', 0)
        
        # Calculate which house the planet is in
        house_position = ((longitude - ascendant + 360) % 360) / 30
        
        # Directional strengths
        # Jupiter & Mercury - 1st house
        # Sun & Mars - 10th house
        # Saturn - 7th house
        # Moon & Venus - 4th house
        
        dig_strength = {
            'Jupiter': 0, 'Mercury': 0, 'Sun': 270, 'Mars': 270,
            'Saturn': 180, 'Moon': 90, 'Venus': 90
        }
        
        if planet in dig_strength:
            ideal_position = dig_strength[planet] / 30
            diff = abs(house_position - ideal_position)
            if diff > 6:
                diff = 12 - diff
            return 60 * (1 - diff / 6)
        
        return 30  # Default
    
    def _calculate_kala_bala(self, planet: str, planet_data: Dict, jd: float) -> float:
        """Calculate Kala Bala (Temporal Strength)"""
        # Simplified calculation
        # In reality includes: Nathonnatha, Paksha, Tribhaga, Abda, Masa, Vara, Hora, Ayana, Yuddha
        
        strength = 0
        
        # Day/Night strength
        # Sun, Jupiter, Venus are day lords (stronger in day)
        # Moon, Mars, Saturn are night lords (stronger at night)
        
        # For simplification, using moderate strength
        if planet in ['Sun', 'Jupiter', 'Venus']:
            strength += 30  # Day strength
        else:
            strength += 30  # Night strength
        
        # Paksha bala (waxing/waning moon)
        if planet == 'Moon':
            strength += 30
        
        return strength
    
    def _calculate_cheshta_bala(self, planet: str, planet_data: Dict) -> float:
        """Calculate Cheshta Bala (Motional Strength)"""
        speed = planet_data.get('speed', 0)
        is_retrograde = planet_data.get('is_retrograde', False)
        
        # Sun and Moon don't have Cheshta Bala
        if planet in ['Sun', 'Moon']:
            return 0
        
        # Retrograde planets get strength
        if is_retrograde:
            return 60
        
        # Fast motion gives strength
        # Simplified - based on speed relative to mean motion
        mean_motions = {
            'Mars': 0.524, 'Mercury': 1.383, 'Jupiter': 0.083,
            'Venus': 1.602, 'Saturn': 0.033
        }
        
        if planet in mean_motions:
            mean = mean_motions[planet]
            ratio = abs(speed) / mean if mean > 0 else 1
            return min(60, 30 * ratio)
        
        return 30
    
    def _calculate_naisargika_bala(self, planet: str) -> float:
        """Calculate Naisargika Bala (Natural Strength)"""
        return self.NATURAL_STRENGTH.get(planet, 30)
    
    def _calculate_drik_bala(self, planet: str, planet_data: Dict, chart_data: Dict) -> float:
        """Calculate Drik Bala (Aspectual Strength)"""
        # Simplified aspect calculation
        # In reality, calculates full aspects from all planets
        
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
            
            # Check for beneficial aspects (trine 120°, opposition 180°, conjunction 0°)
            if abs(diff - 120) < 10:  # Trine
                if planet in self.RELATIONSHIPS and other_name in self.RELATIONSHIPS[planet]['friends']:
                    strength += 15
            elif abs(diff - 180) < 10:  # Opposition
                strength -= 10
            elif diff < 10:  # Conjunction
                if planet in self.RELATIONSHIPS and other_name in self.RELATIONSHIPS[planet]['friends']:
                    strength += 20
                elif planet in self.RELATIONSHIPS and other_name in self.RELATIONSHIPS[planet]['enemies']:
                    strength -= 15
        
        return max(0, strength)
    
    def get_strength_category(self, total_rupas: float) -> str:
        """Categorize planetary strength"""
        if total_rupas >= 6:
            return "Very Strong"
        elif total_rupas >= 5:
            return "Strong"
        elif total_rupas >= 4:
            return "Moderate"
        elif total_rupas >= 3:
            return "Weak"
        else:
            return "Very Weak"