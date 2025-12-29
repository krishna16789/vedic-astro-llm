"""
Tool Calling Framework for Vedic Astrology LLM
Provides tools for chart calculations and analysis
"""

import json
from typing import Dict, Any, List, Callable
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from calculations.core.ephemeris import VedicEphemeris
from calculations.core.shadbala_enhanced import EnhancedShadbalaCalculator
from calculations.core.bhavabala_enhanced import EnhancedBhavabalaCalculator
from calculations.core.dasha import VimshottariDasha
from calculations.core.nakshatra_attributes import NakshatraAttributes


class ToolRegistry:
    """Registry for all available tools"""
    
    def __init__(self):
        self.tools = {}
        self.ephemeris = VedicEphemeris()
        self.shadbala_calc = EnhancedShadbalaCalculator()
        self.bhavabala_calc = EnhancedBhavabalaCalculator()
        self.dasha_calc = VimshottariDasha()
        self.nakshatra_calc = NakshatraAttributes()
        
        # Register all tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools"""
        
        # Chart calculation tools
        self.register_tool(
            name="get_current_chart",
            description="Get the current birth chart data with planetary positions",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=self.get_current_chart
        )
        
        self.register_tool(
            name="calculate_shadbala",
            description="Calculate Shadbala (six-fold strength) for a specific planet or all planets",
            parameters={
                "type": "object",
                "properties": {
                    "planet": {
                        "type": "string",
                        "description": "Planet name (e.g., 'Sun', 'Moon', 'Mars'). If not provided, calculates for all planets.",
                        "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "all"]
                    }
                },
                "required": []
            },
            function=self.calculate_shadbala
        )
        
        self.register_tool(
            name="calculate_bhavabala",
            description="Calculate Bhavabala (house strength) for a specific house or all houses",
            parameters={
                "type": "object",
                "properties": {
                    "house": {
                        "type": "integer",
                        "description": "House number (1-12). If not provided, calculates for all houses.",
                        "minimum": 1,
                        "maximum": 12
                    }
                },
                "required": []
            },
            function=self.calculate_bhavabala
        )
        
        self.register_tool(
            name="get_dasha_periods",
            description="Get Vimshottari Dasha periods (Mahadasha and Antardasha) for current or future time",
            parameters={
                "type": "object",
                "properties": {
                    "years_ahead": {
                        "type": "integer",
                        "description": "Number of years to look ahead (default: 10)",
                        "minimum": 1,
                        "maximum": 120,
                        "default": 10
                    }
                },
                "required": []
            },
            function=self.get_dasha_periods
        )
        
        self.register_tool(
            name="get_nakshatra_attributes",
            description="Get detailed nakshatra attributes including Karakas, Yoni, Gana, Tara for birth chart",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=self.get_nakshatra_attributes
        )
        
        self.register_tool(
            name="get_planet_details",
            description="Get detailed information about a specific planet's position and characteristics",
            parameters={
                "type": "object",
                "properties": {
                    "planet": {
                        "type": "string",
                        "description": "Planet name",
                        "enum": ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
                    }
                },
                "required": ["planet"]
            },
            function=self.get_planet_details
        )
        
        self.register_tool(
            name="get_house_details",
            description="Get detailed information about a specific house including planets and lordship",
            parameters={
                "type": "object",
                "properties": {
                    "house": {
                        "type": "integer",
                        "description": "House number (1-12)",
                        "minimum": 1,
                        "maximum": 12
                    }
                },
                "required": ["house"]
            },
            function=self.get_house_details
        )
        
        self.register_tool(
            name="calculate_divisional_chart",
            description="Calculate positions in a divisional chart (D1, D9, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "varga": {
                        "type": "string",
                        "description": "Varga type (D1, D9, D10, etc.)",
                        "enum": ["D1", "D9", "D10", "D12", "D16", "D20", "D24", "D27", "D30", "D60"]
                    }
                },
                "required": ["varga"]
            },
            function=self.calculate_divisional_chart
        )
    
    def register_tool(self, name: str, description: str, parameters: Dict, function: Callable):
        """Register a new tool"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for LLM"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool = self.tools[tool_name]
        
        try:
            # Update context for tool execution
            self.context = context
            result = tool["function"](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Tool implementations
    def get_current_chart(self) -> Dict[str, Any]:
        """Get current chart data from context"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available. Please calculate a chart first."}
        
        return self.context['chart_data']
    
    def calculate_shadbala(self, planet: str = "all") -> Dict[str, Any]:
        """Calculate Shadbala for planet(s)"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        jd = self.ephemeris.get_julian_day(datetime.fromisoformat(self.context.get('datetime', datetime.now().isoformat())))
        dt = datetime.fromisoformat(self.context.get('datetime', datetime.now().isoformat()))
        
        results = {}
        
        if planet == "all":
            for planet_name, planet_data in chart['planets'].items():
                if planet_name not in ['Rahu', 'Ketu']:
                    planet_with_name = {**planet_data, 'name': planet_name}
                    results[planet_name] = self.shadbala_calc.calculate_shadbala(
                        planet_with_name, chart, jd, dt
                    )
        else:
            if planet not in chart['planets'] or planet in ['Rahu', 'Ketu']:
                return {"error": f"Cannot calculate Shadbala for {planet}"}
            
            planet_data = chart['planets'][planet]
            planet_with_name = {**planet_data, 'name': planet}
            results[planet] = self.shadbala_calc.calculate_shadbala(
                planet_with_name, chart, jd, dt
            )
        
        return results
    
    def calculate_bhavabala(self, house: int = None) -> Dict[str, Any]:
        """Calculate Bhavabala for house(s)"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        jd = self.ephemeris.get_julian_day(datetime.fromisoformat(self.context.get('datetime', datetime.now().isoformat())))
        dt = datetime.fromisoformat(self.context.get('datetime', datetime.now().isoformat()))
        
        # Calculate Shadbala first
        shadbala_results = {}
        for planet_name, planet_data in chart['planets'].items():
            if planet_name not in ['Rahu', 'Ketu']:
                planet_with_name = {**planet_data, 'name': planet_name}
                shadbala_results[planet_name] = self.shadbala_calc.calculate_shadbala(
                    planet_with_name, chart, jd, dt
                )
        
        all_houses = self.bhavabala_calc.calculate_all_houses(chart, shadbala_results)
        
        if house is not None:
            house_key = f"House_{house}"
            if house_key in all_houses:
                return {house_key: all_houses[house_key]}
            else:
                return {"error": f"Invalid house number: {house}"}
        
        return all_houses
    
    def get_dasha_periods(self, years_ahead: int = 10) -> Dict[str, Any]:
        """Get Dasha periods"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        dt = datetime.fromisoformat(self.context.get('datetime', datetime.now().isoformat()))
        moon_longitude = chart['planets']['Moon']['longitude']
        
        current_dasha = self.dasha_calc.get_current_dasha(dt, moon_longitude)
        timeline = self.dasha_calc.get_dasha_timeline(dt, moon_longitude, years_ahead=years_ahead)
        
        return {
            "current_dasha": current_dasha,
            "timeline": timeline[:20]  # Limit to first 20 periods
        }
    
    def get_nakshatra_attributes(self) -> Dict[str, Any]:
        """Get nakshatra attributes"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        return self.nakshatra_calc.get_chart_attributes(chart)
    
    def get_planet_details(self, planet: str) -> Dict[str, Any]:
        """Get detailed planet information"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        
        if planet not in chart['planets']:
            return {"error": f"Planet {planet} not found in chart"}
        
        planet_data = chart['planets'][planet]
        
        # Add additional details
        result = {
            **planet_data,
            "name": planet,
            "sign_lord": self.ephemeris.get_sign_lord(planet_data['sign']),
            "nakshatra_lord": self.ephemeris.get_nakshatra_lord(planet_data['nakshatra'])
        }
        
        return result
    
    def get_house_details(self, house: int) -> Dict[str, Any]:
        """Get detailed house information"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        
        # Calculate house cusp (simple whole sign system)
        asc_sign_num = int(chart['ascendant']['degree'] / 30)
        house_sign_num = (asc_sign_num + house - 1) % 12
        house_sign = self.ephemeris.SIGNS[house_sign_num]
        
        # Find planets in this house
        planets_in_house = []
        for planet_name, planet_data in chart['planets'].items():
            planet_sign_num = int(planet_data['longitude'] / 30)
            if planet_sign_num == house_sign_num:
                planets_in_house.append({
                    "name": planet_name,
                    "degree": planet_data['degree_in_sign']
                })
        
        return {
            "house_number": house,
            "sign": house_sign,
            "sign_lord": self.ephemeris.get_sign_lord(house_sign),
            "planets": planets_in_house
        }
    
    def calculate_divisional_chart(self, varga: str) -> Dict[str, Any]:
        """Calculate divisional chart positions"""
        if not hasattr(self, 'context') or 'chart_data' not in self.context:
            return {"error": "No chart data available"}
        
        chart = self.context['chart_data']
        
        # Calculate divisional positions
        divisional_positions = {}
        
        for planet_name, planet_data in chart['planets'].items():
            divisional_longitude = self.ephemeris.calculate_divisional_chart(
                planet_data['longitude'], varga
            )
            sign_num = int(divisional_longitude / 30)
            
            divisional_positions[planet_name] = {
                "sign": self.ephemeris.SIGNS[sign_num],
                "degree_in_sign": divisional_longitude % 30,
                "longitude": divisional_longitude
            }
        
        # Calculate ascendant in divisional chart
        asc_divisional = self.ephemeris.calculate_divisional_chart(
            chart['ascendant']['degree'], varga
        )
        asc_sign_num = int(asc_divisional / 30)
        
        return {
            "varga": varga,
            "ascendant": {
                "sign": self.ephemeris.SIGNS[asc_sign_num],
                "degree_in_sign": asc_divisional % 30
            },
            "planets": divisional_positions
        }


# Singleton instance
_tool_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get or create tool registry singleton"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry