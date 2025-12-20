#!/usr/bin/env python3
"""
Extract and process training data from VedAstro datasets
Includes horoscope predictions, yogas, planetary data, and event definitions
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
import pandas as pd


class VedAstroExtractor:
    def __init__(self, vedastro_dir: str, output_dir: str):
        self.vedastro_dir = Path(vedastro_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_alpaca_horoscope_data(self) -> List[Dict]:
        """Extract B.V. Raman horoscope predictions from alpaca format"""
        alpaca_file = self.vedastro_dir / "HuggingFace/alpaca_bvraman_horoscope_data.json"
        
        if not alpaca_file.exists():
            print(f"Alpaca file not found: {alpaca_file}")
            return []
        
        with open(alpaca_file, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded {len(data)} horoscope predictions from B.V. Raman")
        return data
    
    def extract_event_definitions(self) -> List[Dict]:
        """Extract astrological event definitions from XML"""
        xml_files = [
            self.vedastro_dir / "Library/XMLData/EventDataList.xml",
            self.vedastro_dir / "Library/XMLData/EventDataList-not-proved.xml"
        ]
        
        all_events = []
        
        for xml_file in xml_files:
            if not xml_file.exists():
                continue
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for event in root.findall('.//Event'):
                    event_data = {
                        "instruction": "Explain this Vedic astrology event",
                        "input": event.find('Name').text if event.find('Name') is not None else "",
                        "output": event.find('Description').text if event.find('Description') is not None else "",
                        "source": "VedAstro_EventDataList",
                        "event_type": event.find('Name').text if event.find('Name') is not None else ""
                    }
                    all_events.append(event_data)
                
                print(f"✓ Extracted {len(all_events)} events from {xml_file.name}")
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
        
        return all_events
    
    def extract_ml_tables(self) -> List[Dict]:
        """Extract planetary data from ML tables"""
        ml_file = self.vedastro_dir / "HuggingFace/ml-table.csv"
        
        if not ml_file.exists():
            print(f"ML table not found: {ml_file}")
            return []
        
        try:
            df = pd.read_csv(ml_file)
            print(f"✓ Loaded ML table with {len(df)} rows")
            
            # Convert to training samples
            samples = []
            for _, row in df.iterrows():
                sample = {
                    "instruction": "Analyze this planetary configuration",
                    "input": json.dumps(row.to_dict()),
                    "output": "Planetary data for astronomical calculations",
                    "source": "VedAstro_MLTable",
                    "data_type": "planetary_positions"
                }
                samples.append(sample)
            
            # Sample only subset for training (too much data otherwise)
            return samples[:1000]  # Take first 1000 samples
        except Exception as e:
            print(f"Error processing ML table: {e}")
            return []
    
    def extract_100_year_data(self) -> List[Dict]:
        """Extract 100 years astronomical data"""
        csv_file = self.vedastro_dir / "HuggingFace/100-years-vedic-astro-london-1900-2000.csv"
        
        if not csv_file.exists():
            print(f"100-year data not found: {csv_file}")
            return []
        
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded 100-year data with {len(df)} records")
            
            # Sample for training
            return []  # This is primarily for calculation validation
        except Exception as e:
            print(f"Error processing 100-year data: {e}")
            return []
    
    def extract_marriage_dataset(self) -> List[Dict]:
        """Extract marriage compatibility data"""
        marriage_file = self.vedastro_dir / "HuggingFace/MarriageInfoDataset.csv"
        
        if not marriage_file.exists():
            print(f"Marriage dataset not found: {marriage_file}")
            return []
        
        try:
            df = pd.read_csv(marriage_file)
            print(f"✓ Loaded marriage dataset with {len(df)} records")
            
            samples = []
            for _, row in df.iterrows():
                sample = {
                    "instruction": "Analyze marriage compatibility",
                    "input": f"Person 1 and Person 2 compatibility analysis",
                    "output": json.dumps(row.to_dict()),
                    "source": "VedAstro_MarriageDataset",
                    "data_type": "marriage_compatibility"
                }
                samples.append(sample)
            
            return samples[:500]  # Sample for training
        except Exception as e:
            print(f"Error processing marriage dataset: {e}")
            return []
    
    def create_yoga_training_data(self, horoscope_data: List[Dict]) -> List[Dict]:
        """Create training samples for yogas (planetary combinations)"""
        yoga_samples = []
        
        # Group by instruction (yoga type)
        yogas = {}
        for item in horoscope_data:
            instruction = item.get('instruction', '')
            if instruction not in yogas:
                yogas[instruction] = []
            yogas[instruction].append(item)
        
        # Create comprehensive yoga explanations
        for yoga_name, samples in yogas.items():
            if len(samples) > 1:
                # Combine multiple interpretations
                combined_output = "\n\n".join([s['output'] for s in samples if s['output']])
                
                yoga_sample = {
                    "instruction": f"Explain the {yoga_name} yoga in Vedic astrology",
                    "input": "",
                    "output": combined_output,
                    "source": "VedAstro_Yogas",
                    "yoga_type": yoga_name,
                    "sample_count": len(samples)
                }
                yoga_samples.append(yoga_sample)
        
        print(f"✓ Created {len(yoga_samples)} yoga training samples")
        return yoga_samples
    
    def process_all_data(self) -> Dict[str, List[Dict]]:
        """Process all VedAstro datasets"""
        print("\n" + "=" * 60)
        print("Extracting VedAstro Training Data")
        print("=" * 60 + "\n")
        
        datasets = {}
        
        # Extract horoscope predictions
        print("1. Extracting horoscope predictions...")
        horoscope_data = self.extract_alpaca_horoscope_data()
        datasets['horoscope_predictions'] = horoscope_data
        
        # Create yoga training data
        print("\n2. Creating yoga training samples...")
        yoga_data = self.create_yoga_training_data(horoscope_data)
        datasets['yogas'] = yoga_data
        
        # Extract event definitions
        print("\n3. Extracting event definitions...")
        event_data = self.extract_event_definitions()
        datasets['events'] = event_data
        
        # Extract ML tables
        print("\n4. Extracting ML table data...")
        ml_data = self.extract_ml_tables()
        datasets['ml_tables'] = ml_data
        
        # Extract marriage data
        print("\n5. Extracting marriage compatibility data...")
        marriage_data = self.extract_marriage_dataset()
        datasets['marriage'] = marriage_data
        
        return datasets
    
    def save_datasets(self, datasets: Dict[str, List[Dict]]):
        """Save all extracted datasets"""
        print("\n" + "=" * 60)
        print("Saving Datasets")
        print("=" * 60 + "\n")
        
        # Save individual datasets
        for name, data in datasets.items():
            if data:
                output_file = self.output_dir / f"vedastro_{name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✓ Saved {len(data)} samples to {output_file}")
        
        # Save combined dataset
        combined = []
        for data in datasets.values():
            combined.extend(data)
        
        output_file = self.output_dir / "vedastro_combined.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Combined dataset: {len(combined)} total samples")
        print(f"✓ Saved to {output_file}")
        
        # Save statistics
        stats = {
            "total_samples": len(combined),
            "datasets": {
                name: len(data) for name, data in datasets.items()
            }
        }
        
        stats_file = self.output_dir / "vedastro_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract training data from VedAstro')
    parser.add_argument(
        '--vedastro-dir',
        default='/Users/krishnatejavemuri/Downloads/VedAstro',
        help='Path to VedAstro directory'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/raw/vedastro',
        help='Output directory for extracted data'
    )
    
    args = parser.parse_args()
    
    extractor = VedAstroExtractor(args.vedastro_dir, args.output_dir)
    datasets = extractor.process_all_data()
    extractor.save_datasets(datasets)
    
    print("\n" + "=" * 60)
    print("VedAstro Data Extraction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()