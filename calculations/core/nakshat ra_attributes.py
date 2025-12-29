"""
Nakshatra Attributes and Karakas
Includes Yoni, Gana, and planetary significators (Karakas)
"""

from typing import Dict, List, Tuple


class NakshatraAttributes:
    """Calculate Nakshatra attributes and Karakas"""
    
    # Nakshatra attributes (Yoni, Gana, etc.)
    NAKSHATRA_DATA = {
        'Ashwini': {'yoni': ('Horse', 'Male'), 'gana': 'Deva', 'lord': 'Ketu'},
        'Bharani': {'yoni': ('Elephant', 'Male'), 'gana': 'Manushya', 'lord': 'Venus'},
        'Krittika': {'yoni': ('Sheep', 'Female'), 'gana': 'Rakshasa', 'lord': 'Sun'},
        'Rohini': {'yoni': ('Serpent', 'Male'), 'gana': 'Manushya', 'lord': 'Moon'},
        'Mrigashira': {'yoni': ('Serpent', 'Female'), 'gana': 'Deva', 'lord': 'Mars'},
        'Ardra': {'yoni': ('Dog', 'Female'), 'gana': 'Manushya', 'lord': 'Rahu'},
        'Punarvasu': {'yoni': ('Cat', 'Female'), 'gana': 'Deva', 'lord': 'Jupiter'},
        'Pushya': {'yoni': ('Sheep', 'Male'), 'gana': 'Deva', 'lord': 'Saturn'},
        'Ashlesha': {'yoni': ('Cat', 'Male'), 'gana': 'Rakshasa', 'lord': 'Mercury'},
        'Magha': {'yoni': ('Rat', 'Male'), 'gana': 'Rakshasa', 'lord': 'Ketu'},
        'Purva Phalguni': {'yoni': ('Rat', 'Female'), 'gana': 'Manushya', 'lord': 'Venus'},
        'Uttara Phalguni': {'yoni': ('Bull', 'Male'), 'gana': 'Manushya', 'lord': 'Sun'},
        'Hasta': {'yoni': ('Buffalo', 'Female'), 'gana': 'Deva', 'lord': 'Moon'},
        'Chitra': {'yoni': ('Tiger', 'Female'), 'gana': 'Rakshasa', 'lord': 'Mars'},
        'Swati': {'yoni': ('Buffalo', 'Male'), 'gana': 'Deva', 'lord': 'Rahu'},
        'Vishakha': {'yoni': ('Tiger', 'Male'), 'gana': 'Rakshasa', 'lord': 'Jupiter'},
        'Anuradha': {'yoni': ('Deer', 'Female'), 'gana': 'Deva', 'lord': 'Saturn'},
        'Jyeshtha': {'yoni': ('Deer', 'Male'), 'gana': 'Rakshasa', 'lord': 'Mercury'},
        'Mula': {'yoni': ('Dog', 'Male'), 'gana': 'Rakshasa', 'lord': 'Ketu'},
        'Purva Ashadha': {'yoni': ('Monkey', 'Male'), 'gana': 'Manushya', 'lord': 'Venus'},
        'Uttara Ashadha': {'yoni': ('Mongoose', 'Male'), 'gana': 'Manushya', 'lord': 'Sun'},
        'Shravana': {'yoni': ('Monkey', 'Female'), 'gana': 'Deva', 'lord': 'Moon'},
        'Dhanishta': {'yoni': ('Lion', 'Female'), 'gana': 'Rakshasa', 'lord': 'Mars'},
        'Shatabhisha': {'yoni': ('Horse', 'Female'), 'gana': 'Rakshasa', 'lord': 'Rahu'},
        'Purva Bhadrapada': {'yoni': ('Lion', 'Male'), 'gana': 'Manushya', 'lord': 'Jupiter'},
        'Uttara Bhadrapada': {'yoni': ('Cow', 'Female'), 'gana': 'Manushya', 'lord': 'Saturn'},
        'Revati': {'yoni': ('Elephant', 'Female'), 'gana': 'Deva', 'lord': 'Mercury'}
    }
    
    # Gana descriptions
    GANA_DESCRIPTIONS = {
        'Deva': 'Divine nature - spiritual, generous, noble',
        'Manushya': 'Human nature - balanced, practical, social',
        'Rakshasa': 'Demonic nature - ambitious, aggressive, materialistic'
    }
    
    def __init__(self):
        """Initialize Nakshatra Attributes calculator"""
        pass
    
    def get_nakshatra_attributes(self, nakshatra: str) -> Dict:
        """Get all attributes for a nakshatra
        
        Args:
            nakshatra: Name of nakshatra
            
        Returns:
            Dictionary with yoni, gana, and lord
        """
        data = self.NAKSHATRA_DATA.get(nakshatra, {})
        if not data:
            return {'error': f'Unknown nakshatra: {nakshatra}'}
        
        yoni_animal, yoni_gender = data.get('yoni', ('Unknown', 'Unknown'))
        gana = data.get('gana', 'Unknown')
        lord = data.get('lord', 'Unknown')
        
        return {
            'nakshatra': nakshatra,
            'yoni': {
                'animal': yoni_animal,
                'gender': yoni_gender,
                'full': f'{yoni_animal} ({yoni_gender})'
            },
            'gana': {
                'type': gana,
                'description': self.GANA_DESCRIPTIONS.get(gana, '')
            },
            'nakshatra_lord': lord
        }
    
    def calculate_karakas(self, chart_data: Dict) -> Dict:
        """Calculate Jaimini Karakas (planetary significators)
        
        Karakas are based on longitudinal degrees in signs (not absolute longitude).
        The planet with highest degree becomes Atmakaraka, next is Amatyakaraka, etc.
        Rahu and Ketu are excluded in some systems but included in others.
        
        Args:
            chart_data: Complete chart data with planets
            
        Returns:
            Dictionary with all Karakas
        """
        planets = chart_data.get('planets', {})
        
        # Get degrees within sign for each planet (excluding or including nodes based on tradition)
        # We'll include all planets including Rahu/Ketu (Jaimini system)
        planet_degrees = {}
        
        for planet_name, planet_data in planets.items():
            degree_in_sign = planet_data.get('degree_in_sign', 0)
            # For Jaimini, we use the degree within the sign
            planet_degrees[planet_name] = degree_in_sign
        
        # Sort planets by degree in descending order
        sorted_planets = sorted(planet_degrees.items(), key=lambda x: x[1], reverse=True)
        
        # Assign Karakas
        karaka_names = [
            'Atmakaraka',      # Self, soul
            'Amatyakaraka',    # Career, minister
            'Bhratrikaraka',   # Siblings, courage
            'Matrikaraka',     # Mother, education
            'Putrakaraka',     # Children, creativity
            'Gnatikaraka',     # Relatives, obstacles
            'Darakaraka',      # Spouse, partnerships
            'Pitrkaraka'       # Father, authority (if 8 planets)
        ]
        
        karakas = {}
        for i, (planet, degree) in enumerate(sorted_planets[:8]):
            if i < len(karaka_names):
                karaka_name = karaka_names[i]
                karakas[karaka_name] = {
                    'planet': planet,
                    'degree_in_sign': round(degree, 2),
                    'significance': self._get_karaka_significance(karaka_name)
                }
        
        return karakas
    
    def _get_karaka_significance(self, karaka: str) -> str:
        """Get the significance of a Karaka"""
        significance = {
            'Atmakaraka': 'Soul, self, main life purpose and direction',
            'Amatyakaraka': 'Career, work, professional life, minister',
            'Bhratrikaraka': 'Siblings, courage, initiatives, younger co-borns',
            'Matrikaraka': 'Mother, education, knowledge, emotional security',
            'Putrakaraka': 'Children, creativity, intelligence, speculation',
            'Gnatikaraka': 'Enemies, obstacles, diseases, relatives',
            'Darakaraka': 'Spouse, partnerships, relationships, marriage',
            'Pitrkaraka': 'Father, authority figures, social status'
        }
        return significance.get(karaka, '')
    
    def get_chart_attributes(self, chart_data: Dict) -> Dict:
        """Get comprehensive attributes for a chart
        
        Args:
            chart_data: Complete chart data
            
        Returns:
            Dictionary with all attributes including karakas and nakshatra info
        """
        planets = chart_data.get('planets', {})
        
        # Get Karakas
        karakas = self.calculate_karakas(chart_data)
        
        # Get attributes for each planet's nakshatra
        nakshatra_attributes = {}
        for planet_name, planet_data in planets.items():
            nakshatra = planet_data.get('nakshatra', '')
            if nakshatra and planet_name not in nakshatra_attributes:
                nakshatra_attributes[planet_name] = self.get_nakshatra_attributes(nakshatra)
        
        # Get Moon's special attributes (most important for personality)
        moon_nakshatra = planets.get('Moon', {}).get('nakshatra', '')
        moon_attributes = self.get_nakshatra_attributes(moon_nakshatra) if moon_nakshatra else {}
        
        return {
            'karakas': karakas,
            'moon_nakshatra_attributes': moon_attributes,
            'all_nakshatra_attributes': nakshatra_attributes,
            'gana_summary': {
                'moon_gana': moon_attributes.get('gana', {}).get('type', 'Unknown'),
                'description': moon_attributes.get('gana', {}).get('description', '')
            },
            'yoni_summary': {
                'moon_yoni': moon_attributes.get('yoni', {}).get('full', 'Unknown'),
                'animal': moon_attributes.get('yoni', {}).get('animal', 'Unknown'),
                'gender': moon_attributes.get('yoni', {}).get('gender', 'Unknown')
            }
        }