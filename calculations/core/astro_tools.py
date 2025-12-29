"""
Astrological Calculation Tools for LLM Function Calling
Provides structured tools that LLM can call to get precise calculations
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .ephemeris import VedicEphemeris
from .shadbala_enhanced import EnhancedShadbalaCalculator
from .bhavabala_enhanced import EnhancedBhavabalaCalculator
from .dasha import VimshottariDasha


class AstroTools:
    """
    Calculation tools accessible to LLM via function calling.
    Each method is a tool the LLM can invoke to get precise astrological data.
    """
    
    def __init__(self):
        self.ephemeris = VedicEphemeris()
        self.shadbala_calc = EnhancedShadbalaCalculator()
        self.bhavabala_calc = EnhancedBhavabalaCalculator()
        self.dasha_calc = VimshottariDasha()
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """
        Return tool definitions for LLM function calling.
        Compatible with OpenAI/Mistral function calling format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_planet_strength",
                    "description": "Calculate Shadbala (six-fold strength) of a planet. Returns total strength and breakdown by source (positional, temporal, etc.). Higher strength means stronger effects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet_name": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"],
                                "description": "Name of the planet to analyze"
                            }
                        },
                        "required": ["planet_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_house_strength",
                    "description": "Calculate Bhavabala (house strength). Indicates the strength of a particular life area. Higher strength means better outcomes in that area.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "house_number": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12,
                                "description": "House number (1-12)"
                            }
                        },
                        "required": ["house_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_navamsa_position",
                    "description": "Get D9 Navamsa chart position of a planet. Navamsa reveals hidden strengths, marriage partner qualities, and destiny.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet_name": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"],
                                "description": "Planet to check in Navamsa"
                            }
                        },
                        "required": ["planet_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_planetary_aspect",
                    "description": "Check if one planet aspects another. Includes special aspects for Mars/Jupiter/Saturn and Rahu/Ketu aspects. Can also check mutual aspects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "aspecting_planet": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"],
                                "description": "Planet casting the aspect"
                            },
                            "aspected_planet": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu", "Ascendant"],
                                "description": "Planet receiving the aspect"
                            },
                            "check_mutual": {
                                "type": "boolean",
                                "description": "Check if planets mutually aspect each other (default: false)"
                            }
                        },
                        "required": ["aspecting_planet", "aspected_planet"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_retrograde_status",
                    "description": "Check if a planet is retrograde. Retrograde planets review past karma and internalize their energies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet_name": {
                                "type": "string",
                                "enum": ["Mars", "Mercury", "Jupiter", "Venus", "Saturn"],
                                "description": "Planet to check (Sun and Moon never retrograde)"
                            }
                        },
                        "required": ["planet_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_combustion",
                    "description": "Check if a planet is combust (too close to Sun). Combust planets lose strength and behave unpredictably.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet_name": {
                                "type": "string",
                                "enum": ["Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"],
                                "description": "Planet to check for combustion"
                            }
                        },
                        "required": ["planet_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_dasha",
                    "description": "Get current Vimshottari Dasha (planetary period). Shows which planet's energy is dominant now and for how long.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "enum": ["mahadasha", "antardasha", "pratyantardasha"],
                                "description": "Level of dasha to retrieve"
                            }
                        },
                        "required": ["level"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_yoga_formation",
                    "description": "Check if a specific yoga (combination) is formed in the chart. Yogas create special effects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "yoga_name": {
                                "type": "string",
                                "description": "Name of yoga to check (e.g., 'Gaja Kesari', 'Panch Mahapurusha', 'Raja Yoga')"
                            }
                        },
                        "required": ["yoga_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_planet_dignity",
                    "description": "Get planet's complete dignity status including exaltation degrees, debilitation, moolatrikona, own signs, and friendly/enemy signs. Includes Rahu/Ketu exaltation points.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet_name": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"],
                                "description": "Planet to check"
                            }
                        },
                        "required": ["planet_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_planetary_friendship",
                    "description": "Check natural friendship between two planets (permanent friends, neutral, enemies). Also shows lordship and temporary friendship based on positions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "planet1": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"],
                                "description": "First planet"
                            },
                            "planet2": {
                                "type": "string",
                                "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"],
                                "description": "Second planet"
                            }
                        },
                        "required": ["planet1", "planet2"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], chart_context: Dict) -> Dict[str, Any]:
        """
        Execute a tool call from LLM.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments from LLM
            chart_context: Current chart data context
            
        Returns:
            Tool execution result
        """
        # Map tool names to methods
        tool_methods = {
            "calculate_planet_strength": self._calc_planet_strength,
            "calculate_house_strength": self._calc_house_strength,
            "get_navamsa_position": self._get_navamsa,
            "check_planetary_aspect": self._check_aspect,
            "check_retrograde_status": self._check_retrograde,
            "check_combustion": self._check_combustion,
            "get_current_dasha": self._get_dasha,
            "check_yoga_formation": self._check_yoga,
            "get_planet_dignity": self._get_dignity,
            "check_planetary_friendship": self._check_friendship
        }
        
        if tool_name not in tool_methods:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return tool_methods[tool_name](arguments, chart_context)
        except Exception as e:
            return {"error": str(e)}
    
    def _calc_planet_strength(self, args: Dict, context: Dict) -> Dict:
        """Calculate Shadbala for a planet"""
        planet_name = args["planet_name"]
        chart = context["chart"]
        jd = context["jd"]
        birth_time = context["birth_time"]
        
        planet_data = chart['planets'].get(planet_name, {})
        planet_with_name = {**planet_data, 'name': planet_name}
        
        shadbala = self.shadbala_calc.calculate_shadbala(planet_with_name, chart, jd, birth_time)
        
        return {
            "planet": planet_name,
            "total_strength": round(shadbala.get('total_strength', 0), 2),
            "strength_sources": {
                "positional": round(shadbala.get('sthana_bala', 0), 2),
                "directional": round(shadbala.get('dig_bala', 0), 2),
                "temporal": round(shadbala.get('kala_bala', 0), 2),
                "motional": round(shadbala.get('chesta_bala', 0), 2),
                "natural": round(shadbala.get('naisargika_bala', 0), 2),
                "aspect": round(shadbala.get('drik_bala', 0), 2)
            },
            "interpretation": self._interpret_strength(shadbala.get('total_strength', 0))
        }
    
    def _calc_house_strength(self, args: Dict, context: Dict) -> Dict:
        """Calculate Bhavabala for a house"""
        house_num = args["house_number"]
        chart = context["chart"]
        
        # Calculate planet strengths first
        shadbala_strengths = {}
        for pname, pdata in chart['planets'].items():
            if pname not in ['Rahu', 'Ketu']:
                pw = {**pdata, 'name': pname}
                sb = self.shadbala_calc.calculate_shadbala(pw, chart, context["jd"], context["birth_time"])
                shadbala_strengths[pname] = sb.get('total_strength', 0)
        
        bhavabala_results = self.bhavabala_calc.calculate_all_houses(chart, shadbala_strengths)
        house_strength = bhavabala_results.get(house_num, {})
        
        return {
            "house": house_num,
            "total_strength": round(house_strength.get('total_strength', 0), 2),
            "interpretation": self._interpret_house_strength(house_strength.get('total_strength', 0))
        }
    
    def _get_navamsa(self, args: Dict, context: Dict) -> Dict:
        """Get Navamsa position"""
        planet_name = args["planet_name"]
        chart = context["chart"]
        
        planet_long = chart['planets'][planet_name]['longitude']
        navamsa_long = self.ephemeris.calculate_divisional_chart(planet_long, 'D9')
        navamsa_sign = self.ephemeris.SIGNS[int(navamsa_long / 30)]
        
        return {
            "planet": planet_name,
            "navamsa_sign": navamsa_sign,
            "degree_in_sign": round(navamsa_long % 30, 2),
            "interpretation": f"{planet_name} in {navamsa_sign} in Navamsa"
        }
    
    def _check_aspect(self, args: Dict, context: Dict) -> Dict:
        """Check planetary aspect including Rahu/Ketu and mutual aspects"""
        asp_planet = args["aspecting_planet"]
        rec_planet = args["aspected_planet"]
        check_mutual = args.get("check_mutual", False)
        chart = context["chart"]
        
        # Get positions
        asp_pos = chart['planets'][asp_planet]['longitude']
        if rec_planet == "Ascendant":
            rec_pos = chart['ascendant']['degree']
        else:
            rec_pos = chart['planets'][rec_planet]['longitude']
        
        # Calculate houses apart
        asp_house = int(asp_pos / 30)
        rec_house = int(rec_pos / 30)
        houses_apart = (rec_house - asp_house) % 12 + 1
        
        # Check special aspects
        is_aspecting = False
        aspect_type = None
        
        # All planets aspect 7th house (opposition)
        if houses_apart == 7:
            is_aspecting = True
            aspect_type = "7th house (opposition)"
        
        # Mars aspects 4th, 7th, 8th
        if asp_planet == "Mars" and houses_apart in [4, 7, 8]:
            is_aspecting = True
            aspect_type = f"{houses_apart}th house (Mars special)"
        
        # Jupiter aspects 5th, 7th, 9th
        if asp_planet == "Jupiter" and houses_apart in [5, 7, 9]:
            is_aspecting = True
            aspect_type = f"{houses_apart}th house (Jupiter special)"
        
        # Saturn aspects 3rd, 7th, 10th
        if asp_planet == "Saturn" and houses_apart in [3, 7, 10]:
            is_aspecting = True
            aspect_type = f"{houses_apart}th house (Saturn special)"
        
        # Rahu/Ketu aspect 5th, 7th, 9th houses
        if asp_planet in ["Rahu", "Ketu"] and houses_apart in [5, 7, 9]:
            is_aspecting = True
            aspect_type = f"{houses_apart}th house ({asp_planet} special)"
        
        result = {
            "aspecting": asp_planet,
            "aspected": rec_planet,
            "is_aspecting": is_aspecting,
            "aspect_type": aspect_type,
            "houses_apart": houses_apart
        }
        
        # Check mutual aspect if requested
        if check_mutual and rec_planet != "Ascendant":
            reverse_houses_apart = (asp_house - rec_house) % 12 + 1
            reverse_aspecting = False
            reverse_type = None
            
            if reverse_houses_apart == 7:
                reverse_aspecting = True
                reverse_type = "7th house (opposition)"
            
            if rec_planet == "Mars" and reverse_houses_apart in [4, 7, 8]:
                reverse_aspecting = True
                reverse_type = f"{reverse_houses_apart}th house (Mars special)"
            
            if rec_planet == "Jupiter" and reverse_houses_apart in [5, 7, 9]:
                reverse_aspecting = True
                reverse_type = f"{reverse_houses_apart}th house (Jupiter special)"
            
            if rec_planet == "Saturn" and reverse_houses_apart in [3, 7, 10]:
                reverse_aspecting = True
                reverse_type = f"{reverse_houses_apart}th house (Saturn special)"
            
            if rec_planet in ["Rahu", "Ketu"] and reverse_houses_apart in [5, 7, 9]:
                reverse_aspecting = True
                reverse_type = f"{reverse_houses_apart}th house ({rec_planet} special)"
            
            result["mutual_aspect"] = {
                "is_mutual": is_aspecting and reverse_aspecting,
                "reverse_aspecting": reverse_aspecting,
                "reverse_type": reverse_type
            }
        
        return result
    
    def _check_retrograde(self, args: Dict, context: Dict) -> Dict:
        """Check if planet is retrograde"""
        planet = args["planet_name"]
        is_retro = context["chart"]['planets'][planet].get('is_retrograde', False)
        
        return {
            "planet": planet,
            "is_retrograde": is_retro,
            "interpretation": f"{planet} is {'retrograde - reviews past karma, internalizes energy' if is_retro else 'direct - expresses normally'}"
        }
    
    def _check_combustion(self, args: Dict, context: Dict) -> Dict:
        """Check if planet is combust"""
        planet = args["planet_name"]
        is_combust = context["chart"]['planets'][planet].get('is_combust', False)
        
        return {
            "planet": planet,
            "is_combust": is_combust,
            "interpretation": f"{planet} is {'combust - weakened by Sun proximity' if is_combust else 'not combust - full strength'}"
        }
    
    def _get_dasha(self, args: Dict, context: Dict) -> Dict:
        """Get current dasha"""
        level = args["level"]
        moon_long = context["chart"]['planets']['Moon']['longitude']
        current_dasha = self.dasha_calc.get_current_dasha(context["birth_time"], moon_long)
        
        return {
            "level": level,
            "current_dasha": current_dasha
        }
    
    def _check_yoga(self, args: Dict, context: Dict) -> Dict:
        """Check yoga formation"""
        yoga_name = args["yoga_name"]
        # TODO: Implement yoga checking logic
        return {
            "yoga": yoga_name,
            "formed": False,
            "note": "Yoga checking not yet implemented"
        }
    
    def _get_dignity(self, args: Dict, context: Dict) -> Dict:
        """Get complete planet dignity with exact degrees"""
        planet = args["planet_name"]
        sign = context["chart"]['planets'][planet]['sign']
        degree = context["chart"]['planets'][planet]['longitude'] % 30
        
        # Complete exaltation data with exact degrees
        exaltation_data = {
            "Sun": ("Aries", 10),
            "Moon": ("Taurus", 3),
            "Mars": ("Capricorn", 28),
            "Mercury": ("Virgo", 15),
            "Jupiter": ("Cancer", 5),
            "Venus": ("Pisces", 27),
            "Saturn": ("Libra", 20),
            "Rahu": ("Gemini", 20),  # Taurus per some texts
            "Ketu": ("Sagittarius", 20)  # Scorpio per some texts
        }
        
        # Debilitation data (exactly opposite exaltation)
        debilitation_data = {
            "Sun": ("Libra", 10),
            "Moon": ("Scorpio", 3),
            "Mars": ("Cancer", 28),
            "Mercury": ("Pisces", 15),
            "Jupiter": ("Capricorn", 5),
            "Venus": ("Virgo", 27),
            "Saturn": ("Aries", 20),
            "Rahu": ("Sagittarius", 20),
            "Ketu": ("Gemini", 20)
        }
        
        # Own signs (lordship/co-ownership)
        own_signs = {
            "Sun": ["Leo"],
            "Moon": ["Cancer"],
            "Mars": ["Aries", "Scorpio"],
            "Mercury": ["Gemini", "Virgo"],
            "Jupiter": ["Sagittarius", "Pisces"],
            "Venus": ["Taurus", "Libra"],
            "Saturn": ["Capricorn", "Aquarius"]
        }
        
        # Moolatrikona signs
        moolatrikona = {
            "Sun": "Leo",
            "Moon": "Taurus",
            "Mars": "Aries",
            "Mercury": "Virgo",
            "Jupiter": "Sagittarius",
            "Venus": "Libra",
            "Saturn": "Aquarius"
        }
        
        # Friend signs (where planet is in friend's sign)
        friend_signs = {
            "Sun": ["Aries", "Sagittarius", "Leo"],  # Mars, Jupiter, Sun
            "Moon": ["Taurus", "Cancer"],  # Venus, Moon
            "Mars": ["Leo", "Sagittarius", "Aries", "Scorpio"],  # Sun, Jupiter, Mars
            "Mercury": ["Gemini", "Virgo", "Taurus", "Libra"],  # Mercury, Venus
            "Jupiter": ["Leo", "Sagittarius", "Pisces", "Cancer"],  # Sun, Jupiter, Moon
            "Venus": ["Gemini", "Virgo", "Taurus", "Libra", "Capricorn", "Aquarius"],  # Mercury, Venus, Saturn
            "Saturn": ["Gemini", "Virgo", "Taurus", "Libra", "Capricorn", "Aquarius"]  # Mercury, Venus, Saturn
        }
        
        # Enemy signs
        enemy_signs = {
            "Sun": ["Taurus", "Libra", "Capricorn", "Aquarius"],  # Venus, Saturn
            "Moon": [],  # Moon has no enemies
            "Mars": ["Gemini", "Virgo"],  # Mercury
            "Mercury": ["Leo", "Sagittarius", "Cancer", "Pisces"],  # Sun, Jupiter, Moon
            "Jupiter": ["Gemini", "Virgo", "Taurus", "Libra"],  # Mercury, Venus
            "Venus": ["Leo", "Aries", "Scorpio"],  # Sun, Mars
            "Saturn": ["Leo", "Aries", "Scorpio", "Cancer"]  # Sun, Mars, Moon
        }
        
        # Determine dignity
        dignity_status = []
        dignity_score = 0
        
        # Check exaltation
        if planet in exaltation_data:
            exalt_sign, exalt_deg = exaltation_data[planet]
            if sign == exalt_sign:
                degree_diff = abs(degree - exalt_deg)
                if degree_diff < 1:
                    dignity_status.append("Deep Exaltation")
                    dignity_score = 100
                else:
                    dignity_status.append("Exalted")
                    dignity_score = 75 + (25 * (1 - degree_diff/30))
        
        # Check debilitation
        if planet in debilitation_data:
            debil_sign, debil_deg = debilitation_data[planet]
            if sign == debil_sign:
                degree_diff = abs(degree - debil_deg)
                if degree_diff < 1:
                    dignity_status.append("Deep Debilitation")
                    dignity_score = -100
                else:
                    dignity_status.append("Debilitated")
                    dignity_score = -75 - (25 * (1 - degree_diff/30))
        
        # Check own sign
        if planet in own_signs and sign in own_signs[planet]:
            dignity_status.append("Own Sign")
            if not dignity_score:  # If not exalted/debilitated
                dignity_score = 60
        
        # Check moolatrikona
        if planet in moolatrikona and sign == moolatrikona[planet]:
            dignity_status.append("Moolatrikona")
            if not dignity_score:
                dignity_score = 55
        
        # Check friend/enemy
        if not dignity_status:  # Only if not exalted/debilitated/own
            if planet in friend_signs and sign in friend_signs[planet]:
                dignity_status.append("Friend Sign")
                dignity_score = 40
            elif planet in enemy_signs and sign in enemy_signs[planet]:
                dignity_status.append("Enemy Sign")
                dignity_score = -40
            else:
                dignity_status.append("Neutral Sign")
                dignity_score = 0
        
        return {
            "planet": planet,
            "sign": sign,
            "degree": round(degree, 2),
            "dignity": dignity_status,
            "dignity_score": round(dignity_score, 2),
            "exaltation_point": exaltation_data.get(planet),
            "debilitation_point": debilitation_data.get(planet),
            "own_signs": own_signs.get(planet, [])
        }
    
    def _interpret_strength(self, strength: float) -> str:
        """Interpret Shadbala strength"""
        if strength > 400:
            return "Very Strong"
        elif strength > 300:
            return "Strong"
        elif strength > 200:
            return "Moderate"
        elif strength > 100:
            return "Weak"
        else:
            return "Very Weak"
    
    def _interpret_house_strength(self, strength: float) -> str:
        """Interpret Bhavabala strength"""
        if strength > 350:
            return "Very Strong - Excellent outcomes"
        elif strength > 250:
            return "Strong - Good outcomes"
        elif strength > 150:
            return "Moderate - Mixed outcomes"
        else:
            return "Weak - Challenging outcomes"
    
    def _check_friendship(self, args: Dict, context: Dict) -> Dict:
        """Check natural and temporary friendship between planets"""
        p1 = args["planet1"]
        p2 = args["planet2"]
        
        # Natural (permanent) friendships
        natural_friends = {
            "Sun": ["Moon", "Mars", "Jupiter"],
            "Moon": ["Sun", "Mercury"],
            "Mars": ["Sun", "Moon", "Jupiter"],
            "Mercury": ["Sun", "Venus"],
            "Jupiter": ["Sun", "Moon", "Mars"],
            "Venus": ["Mercury", "Saturn"],
            "Saturn": ["Mercury", "Venus"]
        }
        
        natural_enemies = {
            "Sun": ["Venus", "Saturn"],
            "Moon": [],
            "Mars": ["Mercury"],
            "Mercury": ["Moon", "Jupiter"],
            "Jupiter": ["Mercury", "Venus"],
            "Venus": ["Sun", "Moon"],
            "Saturn": ["Sun", "Moon", "Mars"]
        }
        
        # Determine natural relationship
        if p2 in natural_friends.get(p1, []):
            natural_rel = "Friend"
        elif p2 in natural_enemies.get(p1, []):
            natural_rel = "Enemy"
        else:
            natural_rel = "Neutral"
        
        # Check if they are co-rulers of same sign
        co_rulers = [
            ["Mars", "Ketu"],  # Scorpio
            ["Jupiter", "Neptune"],  # Pisces (Western)
            ["Saturn", "Uranus"]  # Aquarius (Western)
        ]
        
        are_co_rulers = any((p1 in pair and p2 in pair) for pair in co_rulers)
        
        return {
            "planet1": p1,
            "planet2": p2,
            "natural_relationship": natural_rel,
            "are_co_rulers": are_co_rulers,
            "interpretation": f"{p1} and {p2} are natural {natural_rel.lower()}s"
        }