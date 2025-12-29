"""
Comprehensive Nakshatra Attributes and Karakas
Includes Yoni, Gana, Nadi, Varna, Tattva, Tara Bala, and planetary significators
"""

from typing import Dict, List, Tuple


class NakshatraAttributes:
    """Calculate comprehensive Nakshatra attributes and Karakas"""
    
    # Complete Nakshatra attributes
    NAKSHATRA_DATA = {
        'Ashwini': {
            'yoni': ('Horse', 'Male'), 'gana': 'Deva', 'lord': 'Ketu',
            'nadi': 'Vata', 'varna': 'Vaishya', 'tattva': 'Earth',
            'deity': 'Ashwini Kumaras', 'symbol': 'Horse Head',
            'guna': 'Rajas', 'direction': 'East'
        },
        'Bharani': {
            'yoni': ('Elephant', 'Male'), 'gana': 'Manushya', 'lord': 'Venus',
            'nadi': 'Pitta', 'varna': 'Mleccha', 'tattva': 'Earth',
            'deity': 'Yama', 'symbol': 'Yoni',
            'guna': 'Rajas', 'direction': 'South'
        },
        'Krittika': {
            'yoni': ('Sheep', 'Female'), 'gana': 'Rakshasa', 'lord': 'Sun',
            'nadi': 'Kapha', 'varna': 'Brahmin', 'tattva': 'Fire',
            'deity': 'Agni', 'symbol': 'Razor/Flame',
            'guna': 'Rajas', 'direction': 'North'
        },
        'Rohini': {
            'yoni': ('Serpent', 'Male'), 'gana': 'Manushya', 'lord': 'Moon',
            'nadi': 'Kapha', 'varna': 'Shudra', 'tattva': 'Earth',
            'deity': 'Brahma/Prajapati', 'symbol': 'Chariot/Cart',
            'guna': 'Rajas', 'direction': 'East'
        },
        'Mrigashira': {
            'yoni': ('Serpent', 'Female'), 'gana': 'Deva', 'lord': 'Mars',
            'nadi': 'Pitta', 'varna': 'Servant', 'tattva': 'Earth',
            'deity': 'Soma/Moon', 'symbol': 'Deer Head',
            'guna': 'Tamas', 'direction': 'South'
        },
        'Ardra': {
            'yoni': ('Dog', 'Female'), 'gana': 'Manushya', 'lord': 'Rahu',
            'nadi': 'Vata', 'varna': 'Butcher', 'tattva': 'Water',
            'deity': 'Rudra', 'symbol': 'Teardrop/Diamond',
            'guna': 'Tamas', 'direction': 'West'
        },
        'Punarvasu': {
            'yoni': ('Cat', 'Female'), 'gana': 'Deva', 'lord': 'Jupiter',
            'nadi': 'Vata', 'varna': 'Vaishya', 'tattva': 'Water',
            'deity': 'Aditi', 'symbol': 'Bow/Quiver',
            'guna': 'Rajas', 'direction': 'North'
        },
        'Pushya': {
            'yoni': ('Sheep', 'Male'), 'gana': 'Deva', 'lord': 'Saturn',
            'nadi': 'Pitta', 'varna': 'Kshatriya', 'tattva': 'Water',
            'deity': 'Brihaspati', 'symbol': 'Cow Udder/Lotus',
            'guna': 'Tamas', 'direction': 'East'
        },
        'Ashlesha': {
            'yoni': ('Cat', 'Male'), 'gana': 'Rakshasa', 'lord': 'Mercury',
            'nadi': 'Kapha', 'varna': 'Mleccha', 'tattva': 'Water',
            'deity': 'Nagas', 'symbol': 'Coiled Serpent',
            'guna': 'Sattva', 'direction': 'South'
        },
        'Magha': {
            'yoni': ('Rat', 'Male'), 'gana': 'Rakshasa', 'lord': 'Ketu',
            'nadi': 'Kapha', 'varna': 'Shudra', 'tattva': 'Water',
            'deity': 'Pitris', 'symbol': 'Royal Throne',
            'guna': 'Tamas', 'direction': 'West'
        },
        'Purva Phalguni': {
            'yoni': ('Rat', 'Female'), 'gana': 'Manushya', 'lord': 'Venus',
            'nadi': 'Pitta', 'varna': 'Brahmin', 'tattva': 'Water',
            'deity': 'Bhaga', 'symbol': 'Front Legs of Bed/Hammock',
            'guna': 'Rajas', 'direction': 'North'
        },
        'Uttara Phalguni': {
            'yoni': ('Bull', 'Male'), 'gana': 'Manushya', 'lord': 'Sun',
            'nadi': 'Vata', 'varna': 'Kshatriya', 'tattva': 'Fire',
            'deity': 'Aryaman', 'symbol': 'Back Legs of Bed',
            'guna': 'Rajas', 'direction': 'East'
        },
        'Hasta': {
            'yoni': ('Buffalo', 'Female'), 'gana': 'Deva', 'lord': 'Moon',
            'nadi': 'Vata', 'varna': 'Vaishya', 'tattva': 'Air',
            'deity': 'Savitar/Surya', 'symbol': 'Hand/Fist',
            'guna': 'Rajas', 'direction': 'South'
        },
        'Chitra': {
            'yoni': ('Tiger', 'Female'), 'gana': 'Rakshasa', 'lord': 'Mars',
            'nadi': 'Pitta', 'varna': 'Servant', 'tattva': 'Fire',
            'deity': 'Tvashtar/Vishwakarma', 'symbol': 'Bright Jewel/Pearl',
            'guna': 'Tamas', 'direction': 'West'
        },
        'Swati': {
            'yoni': ('Buffalo', 'Male'), 'gana': 'Deva', 'lord': 'Rahu',
            'nadi': 'Kapha', 'varna': 'Butcher', 'tattva': 'Fire',
            'deity': 'Vayu', 'symbol': 'Young Plant/Coral',
            'guna': 'Tamas', 'direction': 'North'
        },
        'Vishakha': {
            'yoni': ('Tiger', 'Male'), 'gana': 'Rakshasa', 'lord': 'Jupiter',
            'nadi': 'Kapha', 'varna': 'Mleccha', 'tattva': 'Fire',
            'deity': 'Indra-Agni', 'symbol': 'Triumphal Gateway/Archway',
            'guna': 'Sattva', 'direction': 'East'
        },
        'Anuradha': {
            'yoni': ('Deer', 'Female'), 'gana': 'Deva', 'lord': 'Saturn',
            'nadi': 'Pitta', 'varna': 'Shudra', 'tattva': 'Fire',
            'deity': 'Mitra', 'symbol': 'Lotus Flower',
            'guna': 'Tamas', 'direction': 'South'
        },
        'Jyeshtha': {
            'yoni': ('Deer', 'Male'), 'gana': 'Rakshasa', 'lord': 'Mercury',
            'nadi': 'Vata', 'varna': 'Servant', 'tattva': 'Air',
            'deity': 'Indra', 'symbol': 'Circular Amulet/Umbrella',
            'guna': 'Sattva', 'direction': 'West'
        },
        'Mula': {
            'yoni': ('Dog', 'Male'), 'gana': 'Rakshasa', 'lord': 'Ketu',
            'nadi': 'Vata', 'varna': 'Butcher', 'tattva': 'Air',
            'deity': 'Nirriti/Alakshmi', 'symbol': 'Root/Bunch of Roots',
            'guna': 'Tamas', 'direction': 'North'
        },
        'Purva Ashadha': {
            'yoni': ('Monkey', 'Male'), 'gana': 'Manushya', 'lord': 'Venus',
            'nadi': 'Pitta', 'varna': 'Brahmin', 'tattva': 'Air',
            'deity': 'Apas/Varuna', 'symbol': 'Elephant Tusk/Fan',
            'guna': 'Rajas', 'direction': 'East'
        },
        'Uttara Ashadha': {
            'yoni': ('Mongoose', 'Male'), 'gana': 'Manushya', 'lord': 'Sun',
            'nadi': 'Pitta', 'varna': 'Kshatriya', 'tattva': 'Air',
            'deity': 'Vishvadevas', 'symbol': 'Planks of Bed',
            'guna': 'Rajas', 'direction': 'South'
        },
        'Shravana': {
            'yoni': ('Monkey', 'Female'), 'gana': 'Deva', 'lord': 'Moon',
            'nadi': 'Kapha', 'varna': 'Mleccha', 'tattva': 'Air',
            'deity': 'Vishnu', 'symbol': 'Three Footprints/Ear',
            'guna': 'Rajas', 'direction': 'West'
        },
        'Dhanishta': {
            'yoni': ('Lion', 'Female'), 'gana': 'Rakshasa', 'lord': 'Mars',
            'nadi': 'Pitta', 'varna': 'Servant', 'tattva': 'Ether',
            'deity': 'Eight Vasus', 'symbol': 'Drum/Flute',
            'guna': 'Tamas', 'direction': 'North'
        },
        'Shatabhisha': {
            'yoni': ('Horse', 'Female'), 'gana': 'Rakshasa', 'lord': 'Rahu',
            'nadi': 'Vata', 'varna': 'Butcher', 'tattva': 'Ether',
            'deity': 'Varuna', 'symbol': 'Empty Circle/100 Healers',
            'guna': 'Tamas', 'direction': 'East'
        },
        'Purva Bhadrapada': {
            'yoni': ('Lion', 'Male'), 'gana': 'Manushya', 'lord': 'Jupiter',
            'nadi': 'Vata', 'varna': 'Brahmin', 'tattva': 'Ether',
            'deity': 'Aja Ekapada', 'symbol': 'Front Legs of Funeral Cot',
            'guna': 'Sattva', 'direction': 'South'
        },
        'Uttara Bhadrapada': {
            'yoni': ('Cow', 'Female'), 'gana': 'Manushya', 'lord': 'Saturn',
            'nadi': 'Pitta', 'varna': 'Kshatriya', 'tattva': 'Ether',
            'deity': 'Ahir Budhnya', 'symbol': 'Back Legs of Funeral Cot/Twins',
            'guna': 'Tamas', 'direction': 'West'
        },
        'Revati': {
            'yoni': ('Elephant', 'Female'), 'gana': 'Deva', 'lord': 'Mercury',
            'nadi': 'Kapha', 'varna': 'Shudra', 'tattva': 'Ether',
            'deity': 'Pushan', 'symbol': 'Drum/Fish',
            'guna': 'Sattva', 'direction': 'North'
        }
    }
    
    # Tara Bala - compatibility between birth star and current star
    # Count from birth nakshatra to current nakshatra
    TARA_NAMES = [
        'Janma',      # 1, 10, 19 - Birth star, inauspicious
        'Sampat',     # 2, 11, 20 - Wealth, auspicious
        'Vipat',      # 3, 12, 21 - Danger, inauspicious
        'Kshema',     # 4, 13, 22 - Well-being, auspicious
        'Pratyak',    # 5, 14, 23 - Obstacle, inauspicious
        'Sadhana',    # 6, 15, 24 - Achievement, auspicious
        'Naidhana',   # 7, 16, 25 - Death, inauspicious
        'Mitra',      # 8, 17, 26 - Friend, auspicious
        'Parama Mitra' # 9, 18, 27 - Best friend, very auspicious
    ]
    
    TARA_EFFECTS = {
        'Janma': {'result': 'Inauspicious', 'effect': 'Struggles, obstacles'},
        'Sampat': {'result': 'Auspicious', 'effect': 'Wealth, prosperity'},
        'Vipat': {'result': 'Inauspicious', 'effect': 'Danger, losses'},
        'Kshema': {'result': 'Auspicious', 'effect': 'Safety, security'},
        'Pratyak': {'result': 'Inauspicious', 'effect': 'Hurdles, delays'},
        'Sadhana': {'result': 'Auspicious', 'effect': 'Success, achievement'},
        'Naidhana': {'result': 'Inauspicious', 'effect': 'Destruction, death'},
        'Mitra': {'result': 'Auspicious', 'effect': 'Friendship, support'},
        'Parama Mitra': {'result': 'Very Auspicious', 'effect': 'Best results, harmony'}
    }
    
    def __init__(self):
        """Initialize Nakshatra Attributes calculator"""
        pass
    
    def get_nakshatra_attributes(self, nakshatra: str) -> Dict:
        """Get all attributes for a nakshatra"""
        data = self.NAKSHATRA_DATA.get(nakshatra, {})
        if not data:
            return {'error': f'Unknown nakshatra: {nakshatra}'}
        
        yoni_animal, yoni_gender = data.get('yoni', ('Unknown', 'Unknown'))
        
        return {
            'nakshatra': nakshatra,
            'lord': data.get('lord', 'Unknown'),
            'yoni': {
                'animal': yoni_animal,
                'gender': yoni_gender,
                'full': f'{yoni_animal} ({yoni_gender})'
            },
            'gana': data.get('gana', 'Unknown'),
            'nadi': data.get('nadi', 'Unknown'),
            'varna': data.get('varna', 'Unknown'),
            'tattva': data.get('tattva', 'Unknown'),
            'deity': data.get('deity', 'Unknown'),
            'symbol': data.get('symbol', 'Unknown'),
            'guna': data.get('guna', 'Unknown'),
            'direction': data.get('direction', 'Unknown')
        }
    
    def calculate_tara_bala(self, birth_nakshatra: str, current_nakshatra: str) -> Dict:
        """Calculate Tara Bala (star compatibility)
        
        Args:
            birth_nakshatra: Birth nakshatra
            current_nakshatra: Current/transit nakshatra
            
        Returns:
            Tara information
        """
        nakshatras = list(self.NAKSHATRA_DATA.keys())
        
        try:
            birth_index = nakshatras.index(birth_nakshatra)
            current_index = nakshatras.index(current_nakshatra)
        except ValueError:
            return {'error': 'Invalid nakshatra name'}
        
        # Count from birth to current (1-based)
        count = ((current_index - birth_index) % 27) + 1
        
        # Determine Tara (cycles every 9)
        tara_index = (count - 1) % 9
        tara_name = self.TARA_NAMES[tara_index]
        tara_info = self.TARA_EFFECTS[tara_name]
        
        return {
            'birth_nakshatra': birth_nakshatra,
            'current_nakshatra': current_nakshatra,
            'count': count,
            'tara': tara_name,
            'result': tara_info['result'],
            'effect': tara_info['effect']
        }
    
    def calculate_karakas(self, chart_data: Dict) -> Dict:
        """Calculate Jaimini Karakas (planetary significators)"""
        planets = chart_data.get('planets', {})
        planet_degrees = {}
        
        for planet_name, planet_data in planets.items():
            degree_in_sign = planet_data.get('degree_in_sign', 0)
            planet_degrees[planet_name] = degree_in_sign
        
        sorted_planets = sorted(planet_degrees.items(), key=lambda x: x[1], reverse=True)
        
        karaka_names = [
            'Atmakaraka', 'Amatyakaraka', 'Bhratrikaraka', 'Matrikaraka',
            'Putrakaraka', 'Gnatikaraka', 'Darakaraka', 'Pitrkaraka'
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
        """Get comprehensive attributes for a chart"""
        planets = chart_data.get('planets', {})
        
        # Get Karakas
        karakas = self.calculate_karakas(chart_data)
        
        # Get attributes for each planet's nakshatra
        nakshatra_attributes = {}
        for planet_name, planet_data in planets.items():
            nakshatra = planet_data.get('nakshatra', '')
            if nakshatra:
                nakshatra_attributes[planet_name] = self.get_nakshatra_attributes(nakshatra)
        
        # Get Moon's special attributes
        moon_nakshatra = planets.get('Moon', {}).get('nakshatra', '')
        moon_attributes = self.get_nakshatra_attributes(moon_nakshatra) if moon_nakshatra else {}
        
        # Get Ascendant nakshatra
        asc_degree = chart_data.get('ascendant', {}).get('degree', 0)
        asc_nakshatra_num = int(asc_degree / 13.333333333)
        asc_nakshatra_name = list(self.NAKSHATRA_DATA.keys())[asc_nakshatra_num % 27]
        asc_attributes = self.get_nakshatra_attributes(asc_nakshatra_name)
        
        return {
            'karakas': karakas,
            'moon_nakshatra_attributes': moon_attributes,
            'ascendant_nakshatra_attributes': asc_attributes,
            'all_nakshatra_attributes': nakshatra_attributes
        }