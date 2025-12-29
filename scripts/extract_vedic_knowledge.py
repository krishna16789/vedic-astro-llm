"""
Extract Vedic Astrology Knowledge from Classical PDF Books
Extracts structured predictions, yogas, aspects, combinations from classical texts

Data Sources: data/raw/vedastro/books/
Output: Structured JSON knowledge base for LLM interpretation
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Any
import PyPDF2
from collections import defaultdict

class VedicKnowledgeExtractor:
    """Extract and structure Vedic astrology knowledge from PDF books"""
    
    def __init__(self, books_dir: str = "data/raw/vedastro/books"):
        self.books_dir = Path(books_dir)
        self.knowledge_db = {
            "planets_in_houses": defaultdict(list),
            "planets_in_signs": defaultdict(list),
            "planetary_aspects": defaultdict(list),
            "yogas": [],
            "nakshatras": defaultdict(list),
            "divisional_charts": defaultdict(list),
            "retrograde_effects": defaultdict(list),
            "combustion_effects": defaultdict(list),
            "conjunctions": defaultdict(list),
            "source_texts": []
        }
        
        self.planets = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']
        self.signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                     'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean PDF artifacts and excessive whitespace"""
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove page numbers
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    def extract_planet_in_house(self, text: str, source: str):
        """Extract planet in house predictions"""
        for planet in self.planets:
            # Pattern: "Planet in Xth house"
            pattern = rf'{planet}\s+in\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+(?:house|bhava)'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                house = int(match.group(1))
                # Extract prediction (next 500 chars)
                pred_start = match.end()
                pred_end = min(len(text), pred_start + 500)
                prediction = text[pred_start:pred_end].split('.')[0:3]
                
                self.knowledge_db["planets_in_houses"][f"{planet}_H{house}"].append({
                    "planet": planet,
                    "house": house,
                    "prediction": '. '.join(prediction),
                    "source": source
                })
    
    def extract_planet_in_sign(self, text: str, source: str):
        """Extract planet in sign predictions"""
        for planet in self.planets:
            for sign in self.signs:
                pattern = rf'{planet}\s+in\s+{sign}'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    pred_start = match.end()
                    pred_end = min(len(text), pred_start + 500)
                    prediction = text[pred_start:pred_end].split('.')[0:3]
                    
                    self.knowledge_db["planets_in_signs"][f"{planet}_{sign}"].append({
                        "planet": planet,
                        "sign": sign,
                        "prediction": '. '.join(prediction),
                        "source": source
                    })
    
    def extract_aspects(self, text: str, source: str):
        """Extract planetary aspect interpretations"""
        for p1 in self.planets:
            for p2 in self.planets:
                if p1 != p2:
                    pattern = rf'{p1}\s+aspect(?:s|ing)?\s+{p2}'
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        pred_start = match.end()
                        pred_end = min(len(text), pred_start + 400)
                        prediction = text[pred_start:pred_end].split('.')[0:2]
                        
                        self.knowledge_db["planetary_aspects"][f"{p1}_aspects_{p2}"].append({
                            "aspecting": p1,
                            "aspected": p2,
                            "effect": '. '.join(prediction),
                            "source": source
                        })
    
    def extract_yogas(self, text: str, source: str):
        """Extract yoga combinations"""
        yoga_keywords = [
            'gaja kesari', 'budha aditya', 'panch mahapurusha', 'raja yoga',
            'dhana yoga', 'mahabhagya', 'amala', 'viparita', 'neecha bhanga',
            'hamsa', 'malavya', 'sasa', 'ruchaka', 'bhadra', 'adhi yoga',
            'gajakesari', 'sunapha', 'anapha', 'durudhara', 'kemadruma'
        ]
        
        for yoga in yoga_keywords:
            pattern = rf'{yoga}\s+yoga'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pred_start = match.start() - 100
                pred_end = min(len(text), match.end() + 500)
                context = text[max(0, pred_start):pred_end]
                
                self.knowledge_db["yogas"].append({
                    "name": match.group(0),
                    "description": context,
                    "source": source
                })
    
    def extract_nakshatras(self, text: str, source: str):
        """Extract nakshatra predictions"""
        nakshatras = [
            'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra',
            'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'Purva Phalguni', 'Uttara Phalguni',
            'Hasta', 'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha',
            'Mula', 'Purva Ashadha', 'Uttara Ashadha', 'Shravana', 'Dhanishta',
            'Shatabhisha', 'Purva Bhadrapada', 'Uttara Bhadrapada', 'Revati'
        ]
        
        for nakshatra in nakshatras:
            pattern = rf'{nakshatra}\s+nakshatra'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pred_start = match.end()
                pred_end = min(len(text), pred_start + 400)
                prediction = text[pred_start:pred_end].split('.')[0:3]
                
                self.knowledge_db["nakshatras"][nakshatra].append({
                    "nakshatra": nakshatra,
                    "prediction": '. '.join(prediction),
                    "source": source
                })
    
    def extract_retrograde(self, text: str, source: str):
        """Extract retrograde planet effects"""
        for planet in ['Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']:
            pattern = rf'{planet}\s+retrograde'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pred_start = match.end()
                pred_end = min(len(text), pred_start + 400)
                prediction = text[pred_start:pred_end].split('.')[0:2]
                
                self.knowledge_db["retrograde_effects"][planet].append({
                    "planet": planet,
                    "effect": '. '.join(prediction),
                    "source": source
                })
    
    def extract_conjunctions(self, text: str, source: str):
        """Extract conjunction effects"""
        for p1 in self.planets:
            for p2 in self.planets:
                if p1 < p2:  # Avoid duplicates
                    patterns = [
                        rf'{p1}\s+(?:and|with|conjunct)\s+{p2}',
                        rf'{p1}[-\s]{p2}\s+conjunction'
                    ]
                    for pattern in patterns:
                        for match in re.finditer(pattern, text, re.IGNORECASE):
                            pred_start = match.end()
                            pred_end = min(len(text), pred_start + 400)
                            prediction = text[pred_start:pred_end].split('.')[0:2]
                            
                            self.knowledge_db["conjunctions"][f"{p1}_{p2}"].append({
                                "planet1": p1,
                                "planet2": p2,
                                "effect": '. '.join(prediction),
                                "source": source
                            })
    
    def process_all_books(self):
        """Process all PDF books in the directory"""
        if not self.books_dir.exists():
            print(f"Books directory not found: {self.books_dir}")
            return
        
        pdf_files = list(self.books_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            text = self.extract_pdf_text(pdf_path)
            if not text:
                continue
            
            text = self.clean_text(text)
            source = pdf_path.stem
            
            self.knowledge_db["source_texts"].append(source)
            
            # Extract all types of knowledge
            print(f"  - Extracting planet in house predictions...")
            self.extract_planet_in_house(text, source)
            
            print(f"  - Extracting planet in sign predictions...")
            self.extract_planet_in_sign(text, source)
            
            print(f"  - Extracting aspects...")
            self.extract_aspects(text, source)
            
            print(f"  - Extracting yogas...")
            self.extract_yogas(text, source)
            
            print(f"  - Extracting nakshatras...")
            self.extract_nakshatras(text, source)
            
            print(f"  - Extracting retrograde effects...")
            self.extract_retrograde(text, source)
            
            print(f"  - Extracting conjunctions...")
            self.extract_conjunctions(text, source)
    
    def save_knowledge_db(self, output_path: str = "data/processed/vedic_knowledge_db.json"):
        """Save extracted knowledge to JSON"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        db_serializable = {
            "planets_in_houses": dict(self.knowledge_db["planets_in_houses"]),
            "planets_in_signs": dict(self.knowledge_db["planets_in_signs"]),
            "planetary_aspects": dict(self.knowledge_db["planetary_aspects"]),
            "yogas": self.knowledge_db["yogas"],
            "nakshatras": dict(self.knowledge_db["nakshatras"]),
            "divisional_charts": dict(self.knowledge_db["divisional_charts"]),
            "retrograde_effects": dict(self.knowledge_db["retrograde_effects"]),
            "combustion_effects": dict(self.knowledge_db["combustion_effects"]),
            "conjunctions": dict(self.knowledge_db["conjunctions"]),
            "source_texts": self.knowledge_db["source_texts"]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(db_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nKnowledge database saved to: {output_file}")
        self.print_statistics()
    
    def print_statistics(self):
        """Print extraction statistics"""
        print("\n=== Extraction Statistics ===")
        print(f"Source texts: {len(self.knowledge_db['source_texts'])}")
        print(f"Planet in house entries: {sum(len(v) for v in self.knowledge_db['planets_in_houses'].values())}")
        print(f"Planet in sign entries: {sum(len(v) for v in self.knowledge_db['planets_in_signs'].values())}")
        print(f"Aspect entries: {sum(len(v) for v in self.knowledge_db['planetary_aspects'].values())}")
        print(f"Yoga entries: {len(self.knowledge_db['yogas'])}")
        print(f"Nakshatra entries: {sum(len(v) for v in self.knowledge_db['nakshatras'].values())}")
        print(f"Retrograde entries: {sum(len(v) for v in self.knowledge_db['retrograde_effects'].values())}")
        print(f"Conjunction entries: {sum(len(v) for v in self.knowledge_db['conjunctions'].values())}")


def main():
    """Main execution"""
    print("Vedic Astrology Knowledge Extractor")
    print("=" * 50)
    
    extractor = VedicKnowledgeExtractor()
    extractor.process_all_books()
    extractor.save_knowledge_db()
    
    print("\n✅ Extraction complete!")


if __name__ == "__main__":
    main()