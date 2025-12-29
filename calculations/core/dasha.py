"""
Vimshottari Dasha System Calculations
Mahadasha and Antardasha periods based on Moon's nakshatra
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DashaPeriod:
    """Represents a Dasha period"""
    planet: str
    start_date: datetime
    end_date: datetime
    duration_years: float
    
    
class VimshottariDasha:
    """Calculate Vimshottari Dasha system (120-year cycle)"""
    
    # Nakshatra lords in order
    NAKSHATRA_LORDS = [
        'Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter',
        'Saturn', 'Mercury', 'Ketu', 'Venus', 'Sun', 'Moon', 'Mars',
        'Rahu', 'Jupiter', 'Saturn', 'Mercury', 'Ketu', 'Venus', 'Sun',
        'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury'
    ]
    
    # Mahadasha periods in years
    MAHADASHA_YEARS = {
        'Ketu': 7,
        'Venus': 20,
        'Sun': 6,
        'Moon': 10,
        'Mars': 7,
        'Rahu': 18,
        'Jupiter': 16,
        'Saturn': 19,
        'Mercury': 17
    }
    
    # Dasha sequence (starting from any planet, continues in this order)
    DASHA_SEQUENCE = ['Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury']
    
    def __init__(self):
        """Initialize Dasha calculator"""
        pass
    
    def get_birth_nakshatra_lord(self, moon_longitude: float) -> Tuple[str, int]:
        """Get the nakshatra lord at birth based on Moon's position
        
        Args:
            moon_longitude: Moon's longitude in degrees
            
        Returns:
            Tuple of (nakshatra_lord, nakshatra_number)
        """
        # Each nakshatra is 13°20' (13.333...)
        nakshatra_num = int(moon_longitude / 13.333333333)
        nakshatra_lord = self.NAKSHATRA_LORDS[nakshatra_num]
        
        return nakshatra_lord, nakshatra_num
    
    def calculate_balance_of_dasha(self, moon_longitude: float) -> float:
        """Calculate the balance of Mahadasha at birth
        
        Args:
            moon_longitude: Moon's longitude in degrees
            
        Returns:
            Years remaining in birth Mahadasha
        """
        nakshatra_lord, nakshatra_num = self.get_birth_nakshatra_lord(moon_longitude)
        
        # Position within nakshatra (0 to 13.333...)
        position_in_nakshatra = moon_longitude % 13.333333333
        
        # Fraction of nakshatra completed
        fraction_completed = position_in_nakshatra / 13.333333333
        
        # Total years of this Mahadasha
        total_years = self.MAHADASHA_YEARS[nakshatra_lord]
        
        # Balance = years not yet completed
        balance = total_years * (1 - fraction_completed)
        
        return balance
    
    def calculate_mahadasha_periods(self, birth_date: datetime, moon_longitude: float, 
                                   num_periods: int = 9) -> List[DashaPeriod]:
        """Calculate Mahadasha periods from birth
        
        Args:
            birth_date: Date and time of birth
            moon_longitude: Moon's longitude at birth
            num_periods: Number of Mahadasha periods to calculate (default: 9 for one full cycle)
            
        Returns:
            List of DashaPeriod objects
        """
        nakshatra_lord, _ = self.get_birth_nakshatra_lord(moon_longitude)
        balance_years = self.calculate_balance_of_dasha(moon_longitude)
        
        periods = []
        current_date = birth_date
        
        # Find starting position in sequence
        start_index = self.DASHA_SEQUENCE.index(nakshatra_lord)
        
        # First period is the balance of birth Mahadasha
        first_duration_days = int(balance_years * 365.25)
        first_end_date = current_date + timedelta(days=first_duration_days)
        
        periods.append(DashaPeriod(
            planet=nakshatra_lord,
            start_date=current_date,
            end_date=first_end_date,
            duration_years=balance_years
        ))
        
        current_date = first_end_date
        
        # Calculate subsequent Mahadashas
        for i in range(1, num_periods):
            planet_index = (start_index + i) % len(self.DASHA_SEQUENCE)
            planet = self.DASHA_SEQUENCE[planet_index]
            duration_years = self.MAHADASHA_YEARS[planet]
            duration_days = int(duration_years * 365.25)
            end_date = current_date + timedelta(days=duration_days)
            
            periods.append(DashaPeriod(
                planet=planet,
                start_date=current_date,
                end_date=end_date,
                duration_years=duration_years
            ))
            
            current_date = end_date
        
        return periods
    
    def calculate_antardasha_periods(self, mahadasha: DashaPeriod) -> List[DashaPeriod]:
        """Calculate Antardasha (sub-periods) within a Mahadasha
        
        Args:
            mahadasha: The Mahadasha period to subdivide
            
        Returns:
            List of Antardasha periods
        """
        periods = []
        
        # Find starting position in sequence for this Mahadasha
        start_index = self.DASHA_SEQUENCE.index(mahadasha.planet)
        
        current_date = mahadasha.start_date
        total_days = (mahadasha.end_date - mahadasha.start_date).days
        
        # Each Antardasha is proportional to its years in the 120-year cycle
        total_proportional_years = sum(self.MAHADASHA_YEARS.values())
        
        for i in range(len(self.DASHA_SEQUENCE)):
            planet_index = (start_index + i) % len(self.DASHA_SEQUENCE)
            planet = self.DASHA_SEQUENCE[planet_index]
            
            # Antardasha duration is proportional to planet's Mahadasha years
            # within the current Mahadasha period
            proportion = (self.MAHADASHA_YEARS[planet] * mahadasha.duration_years) / total_proportional_years
            duration_days = int((proportion * 365.25))
            
            # Ensure we don't exceed the Mahadasha end date
            end_date = current_date + timedelta(days=duration_days)
            if end_date > mahadasha.end_date:
                end_date = mahadasha.end_date
            
            periods.append(DashaPeriod(
                planet=planet,
                start_date=current_date,
                end_date=end_date,
                duration_years=proportion
            ))
            
            current_date = end_date
            
            if current_date >= mahadasha.end_date:
                break
        
        return periods
    
    def calculate_pratyantardasha_periods(self, antardasha: DashaPeriod) -> List[DashaPeriod]:
        """Calculate Pratyantardasha (sub-sub-periods) within an Antardasha
        
        Args:
            antardasha: The Antardasha period to subdivide
            
        Returns:
            List of Pratyantardasha periods
        """
        periods = []
        
        start_index = self.DASHA_SEQUENCE.index(antardasha.planet)
        current_date = antardasha.start_date
        total_proportional_years = sum(self.MAHADASHA_YEARS.values())
        
        for i in range(len(self.DASHA_SEQUENCE)):
            planet_index = (start_index + i) % len(self.DASHA_SEQUENCE)
            planet = self.DASHA_SEQUENCE[planet_index]
            
            proportion = (self.MAHADASHA_YEARS[planet] * antardasha.duration_years) / total_proportional_years
            duration_days = int((proportion * 365.25))
            
            end_date = current_date + timedelta(days=duration_days)
            if end_date > antardasha.end_date:
                end_date = antardasha.end_date
            
            periods.append(DashaPeriod(
                planet=planet,
                start_date=current_date,
                end_date=end_date,
                duration_years=proportion
            ))
            
            current_date = end_date
            
            if current_date >= antardasha.end_date:
                break
        
        return periods
    
    def get_current_dasha(self, birth_date: datetime, moon_longitude: float,
                         current_date: datetime = None) -> Dict:
        """Get current Mahadasha and Antardasha for a given date
        
        Args:
            birth_date: Date and time of birth
            moon_longitude: Moon's longitude at birth
            current_date: Date to check (default: today)
            
        Returns:
            Dictionary with current Mahadasha, Antardasha, and Pratyantardasha
        """
        if current_date is None:
            current_date = datetime.now()
        
        # Calculate enough Mahadasha periods to cover the current date
        # Maximum 120 years for full Vimshottari cycle
        mahadashas = self.calculate_mahadasha_periods(birth_date, moon_longitude, num_periods=15)
        
        # Find current Mahadasha
        current_mahadasha = None
        for md in mahadashas:
            if md.start_date <= current_date <= md.end_date:
                current_mahadasha = md
                break
        
        if not current_mahadasha:
            return {
                'error': 'Current date is beyond calculated Dasha periods',
                'current_date': current_date.isoformat()
            }
        
        # Calculate Antardashas for current Mahadasha
        antardashas = self.calculate_antardasha_periods(current_mahadasha)
        
        # Find current Antardasha
        current_antardasha = None
        for ad in antardashas:
            if ad.start_date <= current_date <= ad.end_date:
                current_antardasha = ad
                break
        
        # Calculate Pratyantardashas for current Antardasha
        current_pratyantardasha = None
        if current_antardasha:
            pratyantardashas = self.calculate_pratyantardasha_periods(current_antardasha)
            for pd in pratyantardashas:
                if pd.start_date <= current_date <= pd.end_date:
                    current_pratyantardasha = pd
                    break
        
        return {
            'current_date': current_date.isoformat(),
            'mahadasha': {
                'planet': current_mahadasha.planet,
                'start_date': current_mahadasha.start_date.isoformat(),
                'end_date': current_mahadasha.end_date.isoformat(),
                'duration_years': round(current_mahadasha.duration_years, 2)
            },
            'antardasha': {
                'planet': current_antardasha.planet if current_antardasha else None,
                'start_date': current_antardasha.start_date.isoformat() if current_antardasha else None,
                'end_date': current_antardasha.end_date.isoformat() if current_antardasha else None,
                'duration_years': round(current_antardasha.duration_years, 2) if current_antardasha else None
            } if current_antardasha else None,
            'pratyantardasha': {
                'planet': current_pratyantardasha.planet if current_pratyantardasha else None,
                'start_date': current_pratyantardasha.start_date.isoformat() if current_pratyantardasha else None,
                'end_date': current_pratyantardasha.end_date.isoformat() if current_pratyantardasha else None,
                'duration_years': round(current_pratyantardasha.duration_years, 2) if current_pratyantardasha else None
            } if current_pratyantardasha else None
        }
    
    def get_dasha_timeline(self, birth_date: datetime, moon_longitude: float,
                          years_ahead: int = 30) -> Dict:
        """Get a timeline of Mahadasha and Antardasha periods
        
        Args:
            birth_date: Date and time of birth
            moon_longitude: Moon's longitude at birth
            years_ahead: Number of years to calculate ahead
            
        Returns:
            Dictionary with timeline of dashas
        """
        # Calculate Mahadashas
        mahadashas = self.calculate_mahadasha_periods(birth_date, moon_longitude, num_periods=10)
        
        # Filter to only those within the requested timeframe
        end_date = birth_date + timedelta(days=int(years_ahead * 365.25))
        relevant_mahadashas = [md for md in mahadashas if md.start_date <= end_date]
        
        timeline = {
            'birth_date': birth_date.isoformat(),
            'birth_nakshatra_lord': self.get_birth_nakshatra_lord(moon_longitude)[0],
            'balance_at_birth_years': round(self.calculate_balance_of_dasha(moon_longitude), 2),
            'mahadashas': []
        }
        
        for md in relevant_mahadashas:
            # Calculate Antardashas for this Mahadasha
            antardashas = self.calculate_antardasha_periods(md)
            
            md_data = {
                'planet': md.planet,
                'start_date': md.start_date.isoformat(),
                'end_date': md.end_date.isoformat(),
                'duration_years': round(md.duration_years, 2),
                'antardashas': [
                    {
                        'planet': ad.planet,
                        'start_date': ad.start_date.isoformat(),
                        'end_date': ad.end_date.isoformat(),
                        'duration_years': round(ad.duration_years, 2)
                    }
                    for ad in antardashas
                ]
            }
            
            timeline['mahadashas'].append(md_data)
        
        return timeline


# Example usage
if __name__ == "__main__":
    dasha = VimshottariDasha()
    
    # Example birth data
    birth_date = datetime(1990, 1, 1, 10, 30, 0)
    moon_longitude = 125.5  # Example Moon position
    
    # Get current Dasha
    current = dasha.get_current_dasha(birth_date, moon_longitude)
    print("Current Dasha:")
    print(f"Mahadasha: {current['mahadasha']['planet']}")
    if current.get('antardasha'):
        print(f"Antardasha: {current['antardasha']['planet']}")
    if current.get('pratyantardasha'):
        print(f"Pratyantardasha: {current['pratyantardasha']['planet']}")
    
    # Get Dasha timeline
    timeline = dasha.get_dasha_timeline(birth_date, moon_longitude, years_ahead=20)
    print(f"\nBirth Nakshatra Lord: {timeline['birth_nakshatra_lord']}")
    print(f"Balance at birth: {timeline['balance_at_birth_years']} years")