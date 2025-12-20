#!/usr/bin/env python3
"""
Combine and prepare all training data for LLM fine-tuning
Creates train/validation splits and formats for instruction-following
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class TrainingDataPreparator:
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def contains_sanskrit(self, text: str) -> bool:
        """Detect if text contains Sanskrit (Devanagari script)"""
        if not text:
            return False
        
        # Check for Devanagari Unicode characters (U+0900 to U+097F)
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # If more than 5% of characters are Devanagari, consider it Sanskrit
        if len(text) > 0 and (devanagari_count / len(text)) > 0.05:
            return True
        
        return False
        
    def load_all_datasets(self) -> List[Dict]:
        """Load all processed datasets"""
        all_samples = []
        
        # Load VedAstro data
        vedastro_dir = self.raw_data_dir / "vedastro"
        if vedastro_dir.exists():
            for json_file in vedastro_dir.glob("vedastro_*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    all_samples.extend(data)
                    print(f"✓ Loaded {len(data)} samples from {json_file.name}")
        
        # Load book data
        books_dir = self.raw_data_dir / "books"
        if books_dir.exists():
            for json_file in books_dir.glob("*.json"):
                if json_file.name != "extraction_stats.json":
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        all_samples.extend(data)
                        print(f"✓ Loaded {len(data)} samples from {json_file.name}")
        
        print(f"\n✓ Total samples loaded: {len(all_samples)}")
        return all_samples
    
    def clean_and_validate(self, samples: List[Dict]) -> List[Dict]:
        """Clean and validate training samples"""
        cleaned = []
        sanskrit_filtered = 0
        
        for sample in samples:
            # Ensure required fields exist and are not None
            if not all(k in sample for k in ['instruction', 'output']):
                continue
            
            # Skip if any required field is None
            if sample['instruction'] is None or sample['output'] is None:
                continue
            
            # Clean text
            sample['instruction'] = sample['instruction'].strip()
            sample['output'] = sample['output'].strip()
            sample['input'] = sample.get('input', '').strip() if sample.get('input') else ''
            
            # Skip Sanskrit text from books (only keep English)
            if sample.get('source', '').startswith('book_'):
                if self.contains_sanskrit(sample['output']) or self.contains_sanskrit(sample['instruction']):
                    sanskrit_filtered += 1
                    continue
            
            # Skip empty outputs
            if len(sample['output']) < 50:
                continue
            
            # Skip excessively long outputs (token limit)
            if len(sample['output']) > 4000:
                # Truncate long outputs
                sample['output'] = sample['output'][:4000] + "..."
            
            cleaned.append(sample)
        
        print(f"✓ Cleaned samples: {len(cleaned)} (removed {len(samples) - len(cleaned)} invalid)")
        if sanskrit_filtered > 0:
            print(f"  └─ Filtered {sanskrit_filtered} Sanskrit text samples from books")
        return cleaned
    
    def deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples"""
        seen = set()
        unique = []
        
        for sample in samples:
            # Create hash from instruction + output
            key = f"{sample['instruction']}:{sample['output'][:100]}"
            if key not in seen:
                seen.add(key)
                unique.append(sample)
        
        print(f"✓ Deduplicated: {len(unique)} unique (removed {len(samples) - len(unique)} duplicates)")
        return unique
    
    def augment_data(self, samples: List[Dict]) -> List[Dict]:
        """Create augmented versions with rephrased instructions"""
        augmented = samples.copy()
        
        # Instruction variations for common patterns
        variations = {
            "Explain": ["Describe", "What is", "Elaborate on"],
            "What are": ["Explain", "Describe", "What do you know about"],
            "Analyze": ["Examine", "Evaluate", "Interpret"]
        }
        
        for sample in samples[:len(samples)//2]:  # Augment half the dataset
            instruction = sample['instruction']
            
            for original, alternatives in variations.items():
                if instruction.startswith(original):
                    for alt in alternatives[:1]:  # Use first alternative
                        new_sample = sample.copy()
                        new_sample['instruction'] = instruction.replace(original, alt, 1)
                        augmented.append(new_sample)
                        break
        
        print(f"✓ Augmented data: {len(augmented)} total (added {len(augmented) - len(samples)})")
        return augmented
    
    def create_splits(
        self, 
        samples: List[Dict], 
        train_ratio: float = 0.9
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and validation sets"""
        random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        
        print(f"\n✓ Train samples: {len(train_data)}")
        print(f"✓ Validation samples: {len(val_data)}")
        
        return train_data, val_data
    
    def format_for_training(self, samples: List[Dict]) -> List[Dict]:
        """Format samples for instruction-following training"""
        formatted = []
        
        for sample in samples:
            # Alpaca-style format
            if sample.get('input'):
                text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
            else:
                text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
            
            formatted_sample = {
                "text": text,
                "instruction": sample['instruction'],
                "input": sample.get('input', ''),
                "output": sample['output'],
                "source": sample.get('source', 'unknown')
            }
            formatted.append(formatted_sample)
        
        return formatted
    
    def analyze_dataset(self, samples: List[Dict]):
        """Analyze and print dataset statistics"""
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        
        # Count by source
        sources = defaultdict(int)
        for sample in samples:
            sources[sample.get('source', 'unknown')] += 1
        
        print("\nSamples by source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
        
        # Average lengths
        avg_instruction_len = sum(len(s['instruction']) for s in samples) / len(samples)
        avg_output_len = sum(len(s['output']) for s in samples) / len(samples)
        
        print(f"\nAverage instruction length: {avg_instruction_len:.0f} chars")
        print(f"Average output length: {avg_output_len:.0f} chars")
        
        # Token estimates (rough)
        total_tokens = sum(
            len(s['instruction'].split()) + len(s['output'].split()) 
            for s in samples
        )
        print(f"\nEstimated total tokens: {total_tokens:,} (rough estimate)")
    
    def prepare_all(self):
        """Main pipeline to prepare training data"""
        print("=" * 60)
        print("Preparing Training Data for LLM Fine-tuning")
        print("=" * 60 + "\n")
        
        # Load all datasets
        print("1. Loading datasets...")
        all_samples = self.load_all_datasets()
        
        # Clean and validate
        print("\n2. Cleaning and validating...")
        cleaned_samples = self.clean_and_validate(all_samples)
        
        # Deduplicate
        print("\n3. Deduplicating...")
        unique_samples = self.deduplicate(cleaned_samples)
        
        # Augment (optional)
        print("\n4. Augmenting data...")
        augmented_samples = self.augment_data(unique_samples)
        
        # Create splits
        print("\n5. Creating train/validation splits...")
        train_data, val_data = self.create_splits(augmented_samples)
        
        # Format for training
        print("\n6. Formatting for instruction-following...")
        train_formatted = self.format_for_training(train_data)
        val_formatted = self.format_for_training(val_data)
        
        # Analyze
        self.analyze_dataset(train_formatted)
        
        # Save
        print("\n7. Saving datasets...")
        
        train_file = self.output_dir / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_formatted, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved training data: {train_file}")
        
        val_file = self.output_dir / "validation.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_formatted, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved validation data: {val_file}")
        
        # Save metadata
        metadata = {
            "total_samples": len(augmented_samples),
            "train_samples": len(train_formatted),
            "val_samples": len(val_formatted),
            "sources": list(set(s.get('source', 'unknown') for s in train_formatted))
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")
        
        print("\n" + "=" * 60)
        print("Data Preparation Complete!")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for LLM fine-tuning')
    parser.add_argument(
        '--raw-data-dir',
        default='./data/raw',
        help='Directory containing raw extracted data'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/final',
        help='Output directory for final training data'
    )
    
    args = parser.parse_args()
    
    preparator = TrainingDataPreparator(args.raw_data_dir, args.output_dir)
    preparator.prepare_all()


if __name__ == "__main__":
    main()