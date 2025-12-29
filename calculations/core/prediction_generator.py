"""
Vedic Astrology Prediction Generator
Generates weighted predictions from chart data for LLM interpretation
Inspired by VedAstro.org's architecture
"""

from typing import Dict, List, Any
from datetime import datetime
from .ephemeris import VedicEphemeris
from .shadbala_enhanced import EnhancedShadbalaCalculator
from .bhavabala_enhanced import EnhancedBhavabalaCalculator


class PredictionGenerator:
    """
    Generates astrological predictions with weights for LLM interpretation.
    
    Architecture:
    1. Calculate chart positions (planets, houses, nakshatras)
    2. Generate predictions from classical texts
    3. Weight by Shadbala/Bhavabala strength
    4. Format for LLM consumption
    """
    
    def __init__(self):
        self.ephemeris = VedicEphemeris()
        self.shadbala_calc = EnhancedShadbalaCalculator()
        self.bhavabala_calc = EnhancedBhavabalaCalculator()
        
    def generate_predictions(self, birth_time: datetime, lat: float, lon: float) -> List[Dict[str, Any]]:
        """
        Generate all predictions for a birth chart with weights.
        
        Returns list of predictions in VedAstro format:
        [
            {
                "Name": "Sun in 1st House",
                "Description": "Strong personality, leadership qualities...",
                "Weight": 85.3,  # Shadbala/Bhavabala strength
                "Type": "PlanetInHouse",
                "RelatedBodies": ["Sun", "House1"]
            },
            ...
        ]
        """
        predictions = []
        
        # Get chart data
        chart = self.ephemeris.get_chart_data(birth_time, lat, lon)
        jd = self.ephemeris.get_julian_day(birth_time)
        
        # Calculate strengths for weighting
        shadbala_strengths = self._calculate_planet_strengths(chart, jd, birth_time)
        bhavabala_strengths = self._calculate_house_strengths(chart, shadbala_strengths)
        
        # Generate different types of predictions
        predictions.extend(self._planet_in_house_predictions(chart, shadbala_strengths))
        predictions.extend(self._planet_in_sign_predictions(chart, shadbala_strengths))
        predictions.extend(self._nakshatra_predictions(chart, shadbala_strengths))
        predictions.extend(self._house_strength_predictions(chart, bhavabala_strengths))
        predictions.extend(self._planetary_aspect_predictions(chart, shadbala_strengths))
        predictions.extend(self._conjunction_predictions(chart, shadbala_strengths))
        predictions.extend(self._special_yogas(chart, shadbala_strengths))
        
        # Sort by weight (highest first)
        predictions.sort(key=lambda x: x['Weight'], reverse=True)
        
        return predictions
    
    def _calculate_planet_strengths(self, chart: Dict, jd: float, birth_time: datetime) -> Dict[str, float]:
        """Calculate Shadbala strength for all planets"""
        strengths = {}
        for planet_name, planet_data in chart['planets'].items():
            if planet_name not in ['Rahu', 'Ketu']:
                planet_with_name = {**planet_data, 'name': planet_name}
                shadbala = self.shadbala_calc.calculate_shadbala(planet_with_name, chart, jd, birth_time)
                strengths[planet_name] = shadbala.get('total_strength', 0)
        return strengths
    
    def _calculate_house_strengths(self, chart: Dict, shadbala_strengths: Dict) -> Dict[int, float]:
        """Calculate Bhavabala strength for all houses"""
        bhavabala_results = self.bhavabala_calc.calculate_all_houses(chart, shadbala_strengths)
        strengths = {}
        for house_num, house_data in bhavabala_results.items():
            strengths[house_num] = house_data.get('total_strength', 0)
        return strengths
    
    def _planet_in_house_predictions(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions for planets in houses"""
        predictions = []
        asc_sign_num = int(chart['ascendant']['degree'] / 30)
        
        # Classical interpretations from Hindu Predictive Astrology
        interpretations = self._get_planet_house_interpretations()
        
        for planet_name, planet_data in chart['planets'].items():
            planet_sign_num = int(planet_data['longitude'] / 30)
            house_num = ((planet_sign_num - asc_sign_num) % 12) + 1
            
            # Get interpretation
            key = f"{planet_name}_House{house_num}"
            if key in interpretations:
                predictions.append({
                    "Name": f"{planet_name} in House {house_num}",
                    "Description": interpretations[key],
                    "Weight": strengths.get(planet_name, 0),
                    "Type": "PlanetInHouse",
                    "RelatedBodies": [planet_name, f"House{house_num}"]
                })
        
        return predictions
    
    def _planet_in_sign_predictions(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions for planets in signs"""
        predictions = []
        interpretations = self._get_planet_sign_interpretations()
        
        for planet_name, planet_data in chart['planets'].items():
            sign_name = planet_data['sign']
            
            key = f"{planet_name}_{sign_name}"
            if key in interpretations:
                predictions.append({
                    "Name": f"{planet_name} in {sign_name}",
                    "Description": interpretations[key],
                    "Weight": strengths.get(planet_name, 0) * 0.8,  # Slightly less weight than house
                    "Type": "PlanetInSign",
                    "RelatedBodies": [planet_name, sign_name]
                })
        
        return predictions
    
    def _nakshatra_predictions(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions based on nakshatras"""
        predictions = []
        
        # Moon nakshatra is most important
        moon_data = chart['planets'].get('Moon', {})
        moon_nakshatra = moon_data.get('nakshatra', '')
        
        if moon_nakshatra:
            predictions.append({
                "Name": f"Moon in {moon_nakshatra} Nakshatra",
                "Description": self._get_nakshatra_description(moon_nakshatra),
                "Weight": strengths.get('Moon', 0),
                "Type": "Nakshatra",
                "RelatedBodies": ["Moon", moon_nakshatra]
            })
        
        return predictions
    
    def _house_strength_predictions(self, chart: Dict, house_strengths: Dict) -> List[Dict]:
        """Generate predictions based on house strengths"""
        predictions = []
        house_meanings = self._get_house_meanings()
        
        for house_num, strength in house_strengths.items():
            if house_num in house_meanings:
                strength_level = "strong" if strength > 300 else "moderate" if strength > 200 else "weak"
                predictions.append({
                    "Name": f"House {house_num} Strength",
                    "Description": f"House {house_num} ({house_meanings[house_num]}) is {strength_level}. "
                                 f"This affects matters related to {house_meanings[house_num].lower()}.",
                    "Weight": strength * 0.5,  # House strength less direct than planet
                    "Type": "HouseStrength",
                    "RelatedBodies": [f"House{house_num}"]
                })
        
        return predictions
    
    def _planetary_aspect_predictions(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions for planetary aspects"""
        predictions = []
        # TODO: Implement aspect calculations
        # For now, return empty list
        return predictions
    
    def _conjunction_predictions(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions for planetary conjunctions"""
        predictions = []
        asc_sign_num = int(chart['ascendant']['degree'] / 30)
        
        # Group planets by house
        houses = {}
        for planet_name, planet_data in chart['planets'].items():
            planet_sign_num = int(planet_data['longitude'] / 30)
            house_num = ((planet_sign_num - asc_sign_num) % 12) + 1
            
            if house_num not in houses:
                houses[house_num] = []
            houses[house_num].append(planet_name)
        
        # Find conjunctions (2+ planets in same house)
        for house_num, planets in houses.items():
            if len(planets) > 1:
                combined_strength = sum(strengths.get(p, 0) for p in planets) / len(planets)
                planets_str = " and ".join(planets)
                
                predictions.append({
                    "Name": f"Conjunction of {planets_str} in House {house_num}",
                    "Description": f"Conjunction of {planets_str} creates combined energies in house {house_num}. "
                                 f"These planets work together, influencing each other's effects.",
                    "Weight": combined_strength,
                    "Type": "Conjunction",
                    "RelatedBodies": planets + [f"House{house_num}"]
                })
        
        return predictions
    
    def _special_yogas(self, chart: Dict, strengths: Dict) -> List[Dict]:
        """Generate predictions for special yogas"""
        predictions = []
        # TODO: Implement yoga calculations from classical texts
        # For now, return empty list
        return predictions
    
    def _get_planet_house_interpretations(self) -> Dict[str, str]:
        """Classical interpretations from Vedic texts"""
        return {
            "Sun_House1": "Strong personality, leadership qualities, good health, commanding presence. Person of authority.",
            "Sun_House2": "Wealth through government service, eye problems possible, speech is authoritative.",
            "Sun_House10": "Great success in career, recognition from authority, leadership position, fame.",
            "Moon_House1": "Emotional, intuitive, changeable mind, attractive personality, public popularity.",
            "Moon_House4": "Happiness from mother, domestic peace, property ownership, emotional stability.",
            "Mars_House1": "Courageous, aggressive, athletic, prone to accidents, leadership through action.",
            "Mars_House10": "Success through courage, military/police career, technical skills, achievement.",
            "Mercury_House1": "Intelligent, communicative, mathematical abilities, youthful appearance.",
            "Mercury_House10": "Success in business, communication field, writing, intellectual work.",
            "Jupiter_House1": "Wisdom, optimism, good fortune, spiritual inclination, teaching abilities.",
            "Jupiter_House9": "Great fortune, spiritual wisdom, foreign travels, higher education.",
            "Venus_House1": "Beauty, artistic talents, luxury, romantic nature, pleasant personality.",
            "Venus_House7": "Happy marriage, attractive spouse, partnership success, diplomacy.",
            "Saturn_House1": "Serious, disciplined, delayed success, hard work, longevity.",
            "Saturn_House10": "Success through perseverance, authority through time, career stability.",
            # Add more interpretations as needed
        }
    
    def _get_planet_sign_interpretations(self) -> Dict[str, str]:
        """Planet in sign interpretations"""
        return {
            "Sun_Aries": "Leadership, courage, pioneering spirit, strong will, independent nature.",
            "Sun_Leo": "Natural authority, creative, generous, dignified, powerful personality.",
            "Moon_Taurus": "Emotional stability, love of comfort, artistic, possessive nature.",
            "Moon_Cancer": "Highly emotional, nurturing, intuitive, protective, domestic.",
            # Add more interpretations
        }
    
    def _get_nakshatra_description(self, nakshatra: str) -> str:
        """Nakshatra interpretations"""
        descriptions = {
            "Ashwini": "Quick action, healing abilities, pioneering nature, restless energy.",
            "Bharani": "Transformative power, creativity, restraint, nurturing qualities.",
            "Rohini": "Beauty, creativity, material prosperity, emotional depth.",
            # Add all 27 nakshatras
        }
        return descriptions.get(nakshatra, f"Born under {nakshatra} nakshatra.")
    
    def _get_house_meanings(self) -> Dict[int, str]:
        """House significations"""
        return {
            1: "Self, personality, physical body, life path",
            2: "Wealth, family, speech, food",
            3: "Siblings, courage, communication, short travels",
            4: "Mother, home, property, happiness",
            5: "Children, creativity, intelligence, romance",
            6: "Enemies, disease, service, debts",
            7: "Marriage, partnerships, business",
            8: "Longevity, occult, inheritance, transformation",
            9: "Father, fortune, spirituality, higher learning",
            10: "Career, status, public life, achievements",
            11: "Gains, friendships, aspirations, income",
            12: "Loss, liberation, foreign lands, spirituality"
        }
    
    def format_for_llm(self, predictions: List[Dict], top_n: int = 30) -> str:
        """
        Format predictions for LLM consumption (VedAstro style).
        
        Args:
            predictions: List of prediction dictionaries
            top_n: Number of top predictions to include (sorted by weight)
            
        Returns:
            JSON string formatted for LLM
        """
        import json
        
        # Take top N predictions
        top_predictions = predictions[:top_n]
        
        # Format for LLM
        formatted = []
        for pred in top_predictions:
            formatted.append({
                "Name": pred["Name"],
                "Description": pred["Description"],
                "Weight": round(pred["Weight"], 2),
                "Type": pred["Type"]
            })
        
        return json.dumps(formatted, indent=2)