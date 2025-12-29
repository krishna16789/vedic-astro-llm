"""
Bhavabala (House Strength) Calculations
Based on Parasara's classical Vedic Astrology principles
"""

import math
from typing import Dict, List
import swisseph as swe


class BhavabalaCalculator:
    """Calculate Bhavabala (House Strength)"""
    
    # Signs ruled by planets
    SIGN_RULERS = {
        0: 'Mars',      # Aries
        1: 'Venus',     # Taurus
        2: 'Mercury',   # Gemini
        3: 'Moon',      # Cancer
        4: 'Sun',       # Leo
        5: 'Mercury',   # Virgo
        6: 'Venus',     # Libra
        7: 'Mars',      # Scorpio
        8: 'Jupiter',   # Sagittarius
        9: 'Saturn',    # Capricorn
        10: 'Saturn',   # Aquarius
        11: 'Jupiter'   # Pisces
    }
    
    def __init__(self):
        """Initialize Bhavabala calculator"""
        swe.set_sid_mode(swe.SIDM_LAHIRI)
    
    def calculate_bhavabala(self, house_num: int, house_cusp: float, 
                           chart_data: Dict, shadbala_results: Dict = None) -> Dict:
        """Calculate Bhavabala for a specific house
        
        Args:
            house_num: House number (1-12)
            house_cusp: House cusp longitude
            chart_data: Complete chart data
            shadbala_results: Pre-calculated Shadbala results for planets
            
        Returns:
            Dictionary with Bhavabala components
        """
        # Calculate five components of Bhavabala
        bhavadhipati_bala = self._calculate_bhavadhipati_bala(
            house_num, house_cusp, chart_data, shadbala_results
        )
        bhava_dig_bala = self._calculate_bhava_dig_bala(house_num)
        bhava_drishti_bala = self._calculate_bhava_drishti_bala(
            house_num, house_cusp, chart_data
        )
        bhava_occupied_bala = self._calculate_bhava_occupied_bala(
            house_num, house_cusp, chart_data
        )
        bhava_nature_bala = self._calculate_bhava_nature_bala(house_num)
        
        total = (bhavadhipati_bala + bhava_dig_bala + bhava_drishti_bala + 
                bhava_occupied_bala + bhava_nature_bala)
        
        return {
            'house': house_num,
            'total': total,
            'rupas': total / 60,
            'strength_category': self._get_strength_category(total / 60),
            'components': {
                'bhavadhipati_bala': bhavadhipati_bala,
                'bhava_dig_bala': bhava_dig_bala,
                'bhava_drishti_bala': bhava_drishti_bala,
                'bhava_occupied_bala': bhava_occupied_bala,
                'bhava_nature_bala': bhava_nature_bala
            }
        }
    
    def _calculate_bhavadhipati_bala(self, house_num: int, house_cusp: float,
                                     chart_data: Dict, shadbala_results: Dict) -> float:
        """Calculate strength based on house lord's Shadbala"""
        sign_num = int(house_cusp / 30)
        lord = self.SIGN_RULERS.get(sign_num)
        
        if not lord:
            return 30  # Default
        
        # If Shadbala results provided, use them
        if shadbala_results and lord in shadbala_results:
            return shadbala_results[lord].get('total', 30)
        
        # Otherwise, return moderate strength
        return 30
    
    def _calculate_bhava_dig_bala(self, house_num: int) -> float:
        """Calculate directional strength of house
        
        Angular houses (1, 4, 7, 10) are strongest
        Succedent houses (2, 5, 8, 11) are moderate
        Cadent houses (3, 6, 9, 12) are weakest
        """
        if house_num in [1, 4, 7, 10]:  # Angular (Kendra)
            return 60
        elif house_num in [2, 5, 8, 11]:  # Succedent (Panapara)
            return 40
        else:  # Cadent (Apoklima) - 3, 6, 9, 12
            return 20
    
    def _calculate_bhava_drishti_bala(self, house_num: int, house_cusp: float,
                                      chart_data: Dict) -> float:
        """Calculate strength from planetary aspects on the house"""
        strength = 0
        planets = chart_data.get('planets', {})
        
        for planet_name, planet_data in planets.items():
            if planet_name in ['Rahu', 'Ketu']:
                continue
            
            planet_long = planet_data.get('longitude', 0)
            
            # Calculate angular difference
            diff = abs(house_cusp - planet_long)
            if diff > 180:
                diff = 360 - diff
            
            # Check for aspects
            # 7th aspect (opposition) - all planets
            if abs(diff - 180) < 15:
                strength += 20
            
            # 4th and 8th aspects - Mars
            if planet_name == 'Mars':
                if abs(diff - 90) < 15 or abs(diff - 270) < 15:
                    strength += 15
            
            # 5th and 9th aspects - Jupiter
            if planet_name == 'Jupiter':
                if abs(diff - 120) < 15 or abs(diff - 240) < 15:
                    strength += 20
            
            # 3rd and 10th aspects - Saturn
            if planet_name == 'Saturn':
                if abs(diff - 60) < 15 or abs(diff - 270) < 15:
                    strength += 15
        
        return min(60, strength)  # Cap at 60
    
    def _calculate_bhava_occupied_bala(self, house_num: int, house_cusp: float,
                                       chart_data: Dict) -> float:
        """Calculate strength based on planets occupying the house"""
        strength = 0
        planets = chart_data.get('planets', {})
        house_end = house_cusp + 30
        
        # Count benefics and malefics in house
        benefics = ['Jupiter', 'Venus', 'Mercury', 'Moon']
        malefics = ['Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu']
        
        benefic_count = 0
        malefic_count = 0
        
        for planet_name, planet_data in planets.items():
            planet_long = planet_data.get('longitude', 0)
            
            # Check if planet is in this house
            if house_cusp <= planet_long < house_end:
                if planet_name in benefics:
                    benefic_count += 1
                    strength += 15
                elif planet_name in malefics:
                    malefic_count += 1
                    # Malefics in 3, 6, 10, 11 houses are good
                    if house_num in [3, 6, 10, 11]:
                        strength += 10
                    else:
                        strength -= 5
        
        return max(0, min(60, strength))
    
    def _calculate_bhava_nature_bala(self, house_num: int) -> float:
        """Calculate strength based on natural significance of house
        
        Trikona houses (1, 5, 9) - Most auspicious
        Kendra houses (1, 4, 7, 10) - Powerful
        Dusthana houses (6, 8, 12) - Challenging
        """
        if house_num in [1, 5, 9]:  # Trikona (Trines)
            return 50
        elif house_num in [4, 7, 10]:  # Kendra (not already in trikona)
            return 40
        elif house_num in [2, 11]:  # Dhana (wealth) houses
            return 35
        elif house_num in [3]:  # Upachaya (growing)
            return 30
        elif house_num in [6, 8, 12]:  # Dusthana (challenging)
            return 15
        else:
            return 25
    
    def _get_strength_category(self, rupas: float) -> str:
        """Categorize house strength"""
        if rupas >= 5:
            return "Very Strong"
        elif rupas >= 4:
            return "Strong"
        elif rupas >= 3:
            return "Moderate"
        elif rupas >= 2:
            return "Weak"
        else:
            return "Very Weak"
    
    def calculate_all_houses(self, chart_data: Dict, shadbala_results: Dict = None) -> Dict:
        """Calculate Bhavabala for all 12 houses"""
        house_cusps = chart_data.get('house_cusps', [])
        
        if len(house_cusps) < 12:
            return {'error': 'Insufficient house cusp data'}
        
        results = {}
        for i in range(12):
            house_num = i + 1
            house_cusp = house_cusps[i]
            results[house_num] = self.calculate_bhavabala(
                house_num, house_cusp, chart_data, shadbala_results
            )
        
        return results