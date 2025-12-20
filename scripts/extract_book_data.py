#!/usr/bin/env python3
"""
Extract text from Vedic astrology books (PDFs with searchable text)
Avoids OCR by only processing books with embedded text
Uses PyMuPDF (fitz) for better text extraction quality
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import re

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    import PyPDF2
    print("⚠️  PyMuPDF not found, falling back to PyPDF2 (lower quality)")
    print("   Install with: pip install pymupdf")


class BookExtractor:
    def __init__(self, books_dir: str, output_dir: str):
        self.books_dir = Path(books_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def is_searchable_pdf(self, pdf_path: Path) -> bool:
        """Check if PDF has searchable text (not scanned image)"""
        try:
            if HAS_PYMUPDF:
                # PyMuPDF method
                doc = fitz.open(pdf_path)
                # Check first 5 pages for text
                for i in range(min(5, len(doc))):
                    text = doc[i].get_text()
                    if len(text.strip()) > 100:  # Has substantial text
                        doc.close()
                        return True
                doc.close()
                return False
            else:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for i in range(min(5, len(reader.pages))):
                        text = reader.pages[i].extract_text()
                        if len(text.strip()) > 100:
                            return True
                    return False
        except Exception as e:
            print(f"Error checking {pdf_path}: {e}")
            return False
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract all text from a searchable PDF using best available method"""
        try:
            if HAS_PYMUPDF:
                # PyMuPDF method - much better for complex PDFs
                doc = fitz.open(pdf_path)
                text = ""
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Extract text with layout preservation
                    page_text = page.get_text("text", sort=True)
                    text += page_text + "\n\n"
                doc.close()
                return text
            else:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n\n"
                    return text
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
            return ""
    
    def is_corrupted_or_garbled(self, text: str) -> bool:
        """
        Detect if text is corrupted, garbled, or unreadable (OCR errors, encoding issues)
        
        Args:
            text: Text to analyze
        
        Returns:
            True if text appears corrupted or unreadable
        """
        if not text or len(text.strip()) < 50:
            return True
        
        # Count various character types
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        
        # Check for excessive special characters (corrupted text indicator)
        if total_chars > 0:
            special_ratio = special_chars / total_chars
            if special_ratio > 0.4:  # >40% special characters
                return True
        
        # Check for excessive digits mixed with text (table/corrupted data)
        if alpha_chars > 0:
            digit_ratio = digit_chars / (alpha_chars + digit_chars)
            if digit_ratio > 0.4:  # >40% digits in alphanumeric content
                return True
        
        # Check for very low readable English content
        words = text.split()
        if len(words) > 10:
            # Count words that look like English (mostly ASCII letters)
            english_like = sum(1 for word in words
                             if len(word) > 2 and
                             sum(1 for c in word if c.isascii() and c.isalpha()) / len(word) > 0.7)
            english_ratio = english_like / len(words)
            
            # If less than 30% looks like English and has Devanagari, it's probably corrupted
            if english_ratio < 0.3 and devanagari_chars > 0:
                return True
        
        # Check for patterns indicating corruption: excessive punctuation clusters
        corruption_patterns = ['॥॥', '।।', '..', ',,', ';;', '  \\', '\\/', '^^']
        corruption_count = sum(text.count(pattern) for pattern in corruption_patterns)
        if corruption_count > 5:
            return True
        
        return False
    
    def is_sanskrit_chunk(self, text: str, threshold: float = 0.15) -> bool:
        """
        Detect if text chunk contains significant Sanskrit/Devanagari content
        
        Args:
            text: Text to analyze
            threshold: Minimum ratio of Devanagari characters to consider Sanskrit (default 0.15)
        
        Returns:
            True if chunk appears to be primarily Sanskrit
        """
        if not text or len(text.strip()) == 0:
            return False
        
        # Count Devanagari Unicode characters (U+0900 to U+097F)
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # Calculate ratio of Devanagari to total characters (excluding whitespace)
        total_chars = len([c for c in text if not c.isspace()])
        
        if total_chars == 0:
            return False
        
        devanagari_ratio = devanagari_count / total_chars
        
        # Lower threshold (15%) to catch more Sanskrit/mixed content
        return devanagari_ratio > threshold
    
    def is_irrelevant_chunk(self, text: str) -> bool:
        """
        Detect if chunk is irrelevant metadata (copyright, dedication, preface, etc.)
        
        Args:
            text: Text to analyze
        
        Returns:
            True if chunk appears to be irrelevant metadata
        """
        if not text or len(text.strip()) < 50:  # Too short to be meaningful
            return True
        
        text_lower = text.lower()
        
        # Keywords indicating front matter or non-instructional content
        irrelevant_keywords = [
            'copyright', '©', 'all rights reserved', 'published by',
            'dedication', 'dedicated to', 'acknowledgment', 'acknowledgement',
            'preface', 'foreword', 'introduction to this edition',
            'about the author', 'about this book', 'how to use this book',
            'table of contents', 'contents', 'index',
            'first published', 'isbn', 'printed in',
            'looking back: an update', 'in the last ten years',
            'http://', 'https://', 'www.', '@', 'email:',
            'page intentionally left blank', 'this page intentionally'
        ]
        
        # Check if chunk contains multiple irrelevant keywords
        keyword_count = sum(1 for keyword in irrelevant_keywords if keyword in text_lower)
        if keyword_count >= 2:
            return True
        
        # Check for URL patterns (more robust than just http://)
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        urls = re.findall(url_pattern, text)
        if len(urls) > 2:  # Multiple URLs suggest metadata
            return True
        
        # Check if it's mostly a list of names (dedication/acknowledgment pattern)
        lines = text.split('\n')
        short_lines = [l for l in lines if 0 < len(l.strip()) < 50]
        if len(short_lines) > len(lines) * 0.7:  # >70% short lines
            return True
        
        # Check for copyright symbol or year patterns typical of copyright pages
        if '©' in text or 'first published' in text_lower:
            return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 2000, filter_sanskrit: bool = True,
                   filter_irrelevant: bool = True, filter_corrupted: bool = True) -> List[str]:
        """
        Split text into manageable chunks and filter unwanted content
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            filter_sanskrit: Whether to filter out Sanskrit chunks (default True)
            filter_irrelevant: Whether to filter irrelevant metadata (default True)
            filter_corrupted: Whether to filter corrupted/garbled text (default True)
        
        Returns:
            List of text chunks with unwanted content removed
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        sanskrit_filtered = 0
        irrelevant_filtered = 0
        corrupted_filtered = 0
        
        for para in paragraphs:
            # Skip corrupted/garbled paragraphs first (most important)
            if filter_corrupted and self.is_corrupted_or_garbled(para):
                corrupted_filtered += 1
                continue
            
            # Skip Sanskrit paragraphs if filtering is enabled
            if filter_sanskrit and self.is_sanskrit_chunk(para):
                sanskrit_filtered += 1
                continue
            
            # Skip irrelevant metadata paragraphs
            if filter_irrelevant and self.is_irrelevant_chunk(para):
                irrelevant_filtered += 1
                continue
                
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    # Double-check the complete chunk
                    chunk_to_check = current_chunk.strip()
                    is_corrupted = filter_corrupted and self.is_corrupted_or_garbled(chunk_to_check)
                    is_sanskrit = filter_sanskrit and self.is_sanskrit_chunk(chunk_to_check)
                    is_irrelevant = filter_irrelevant and self.is_irrelevant_chunk(chunk_to_check)
                    
                    if not (is_corrupted or is_sanskrit or is_irrelevant):
                        chunks.append(chunk_to_check)
                    else:
                        if is_corrupted:
                            corrupted_filtered += 1
                        if is_sanskrit:
                            sanskrit_filtered += 1
                        if is_irrelevant:
                            irrelevant_filtered += 1
                            
                current_chunk = para + "\n\n"
        
        if current_chunk:
            # Double-check the final chunk
            chunk_to_check = current_chunk.strip()
            is_corrupted = filter_corrupted and self.is_corrupted_or_garbled(chunk_to_check)
            is_sanskrit = filter_sanskrit and self.is_sanskrit_chunk(chunk_to_check)
            is_irrelevant = filter_irrelevant and self.is_irrelevant_chunk(chunk_to_check)
            
            if not (is_corrupted or is_sanskrit or is_irrelevant):
                chunks.append(chunk_to_check)
            else:
                if is_corrupted:
                    corrupted_filtered += 1
                if is_sanskrit:
                    sanskrit_filtered += 1
                if is_irrelevant:
                    irrelevant_filtered += 1
        
        # Report filtering statistics
        if corrupted_filtered > 0:
            print(f"  ℹ️  Filtered {corrupted_filtered} corrupted/garbled chunks")
        if sanskrit_filtered > 0:
            print(f"  ℹ️  Filtered {sanskrit_filtered} Sanskrit chunks")
        if irrelevant_filtered > 0:
            print(f"  ℹ️  Filtered {irrelevant_filtered} irrelevant/metadata chunks")
        
        return chunks
    
    def extract_training_samples(self, text: str, book_name: str,
                                filter_sanskrit: bool = True,
                                filter_irrelevant: bool = True,
                                filter_corrupted: bool = True) -> List[Dict]:
        """
        Convert text chunks into training samples
        
        Args:
            text: Text to process
            book_name: Name of the source book
            filter_sanskrit: Whether to filter Sanskrit chunks (default True)
            filter_irrelevant: Whether to filter irrelevant metadata (default True)
            filter_corrupted: Whether to filter corrupted/garbled text (default True)
        
        Returns:
            List of training samples with unwanted content filtered
        """
        chunks = self.chunk_text(text, filter_sanskrit=filter_sanskrit,
                                filter_irrelevant=filter_irrelevant,
                                filter_corrupted=filter_corrupted)
        samples = []
        
        for i, chunk in enumerate(chunks):
            # Create instruction-following format
            sample = {
                "instruction": f"Explain the following Vedic astrology concept from {book_name}",
                "input": "",
                "output": chunk,
                "source": book_name,
                "chunk_id": i
            }
            samples.append(sample)
        
        return samples
    
    def process_books(self) -> Dict[str, List[Dict]]:
        """Process all books in the directory"""
        all_samples = {}
        
        # Priority books (known good quality)
        priority_books = [
            "2015.83552.Three-Hundred-Important-Combinations.pdf",
            "2015.312156.Jataka-Parijata.pdf",
            "2015.406251.Brihat-Jataka_text.pdf",
            "Brihat Parasara Hora Sastra with English Translation Girish Chand Sharma Volume 1.pdf",
            "Brihat Parasara Hora Sastra with English Translation Girish Chand Sharma Volume 2.pdf",
            "Mantreswara_s__Phaladeeplka_.pdf",
            "Uttara Kalamritam.pdf",
            "Astrology For The Soul PDF.pdf",
            "Bepin Behari_Fundamentals of Vedic Astrology.pdf",
            "Light on Life_ An Introduction to the Astrology of India - PDF Room.pdf",
            "vedic_astro_textbook.pdf"
        ]
        
        for book_name in priority_books:
            book_path = self.books_dir / book_name
            
            if not book_path.exists():
                print(f"Book not found: {book_name}")
                continue
            
            print(f"\nProcessing: {book_name}")
            
            # Check if searchable
            if not self.is_searchable_pdf(book_path):
                print(f"  ⚠️  Skipping (requires OCR): {book_name}")
                continue
            
            # Extract text
            print(f"  ✓ Extracting text...")
            text = self.extract_pdf_text(book_path)
            
            if len(text) < 1000:
                print(f"  ⚠️  Insufficient text extracted: {book_name}")
                continue
            
            # Clean and chunk
            text = self.clean_text(text)
            print(f"  ✓ Extracted {len(text)} characters")
            
            # Create training samples
            samples = self.extract_training_samples(text, book_name)
            print(f"  ✓ Created {len(samples)} training samples")
            
            all_samples[book_name] = samples
            
            # Save individual book data
            output_file = self.output_dir / f"{Path(book_name).stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to {output_file}")
        
        return all_samples
    
    def save_combined_dataset(self, all_samples: Dict[str, List[Dict]]):
        """Save all samples into a single training dataset"""
        combined = []
        for book_name, samples in all_samples.items():
            combined.extend(samples)
        
        output_file = self.output_dir / "combined_training_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Combined dataset: {len(combined)} samples")
        print(f"✓ Saved to {output_file}")
        
        # Save statistics
        stats = {
            "total_samples": len(combined),
            "books_processed": len(all_samples),
            "samples_per_book": {
                book: len(samples)
                for book, samples in all_samples.items()
            },
            "filtering": {
                "corrupted_text_filtered": True,
                "sanskrit_filtered": True,
                "metadata_filtered": True,
                "note": "Corrupted/garbled text, Sanskrit content, and irrelevant metadata filtered out"
            }
        }
        
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract training data from astrology books')
    parser.add_argument(
        '--books-dir',
        default='/Users/krishnatejavemuri/Downloads/astrology books',
        help='Directory containing astrology books'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/raw/books',
        help='Output directory for extracted data'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vedic Astrology Book Data Extraction")
    print("=" * 60)
    
    extractor = BookExtractor(args.books_dir, args.output_dir)
    all_samples = extractor.process_books()
    extractor.save_combined_dataset(all_samples)
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()