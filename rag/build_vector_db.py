"""
Build Vector Database for RAG (Retrieval Augmented Generation)
Creates embeddings from all Vedic astrology books and stores them for quick retrieval
"""

import json
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import pickle

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

class VectorDatabase:
    """Vector database for semantic search over Vedic astrology texts"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector database
        
        Args:
            model_name: Sentence transformer model name
                       'all-MiniLM-L6-v2' - Fast, 384 dimensions, ~80MB
                       'all-mpnet-base-v2' - Better quality, 768 dimensions, ~420MB
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.metadata = []
        
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending in next 100 chars
                sentence_end = text[end:end+100].find('. ')
                if sentence_end != -1:
                    end += sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def load_book_data(self, book_path: str) -> List[Dict[str, Any]]:
        """
        Load book data from JSON file
        
        Args:
            book_path: Path to book JSON file
            
        Returns:
            List of text entries with metadata
        """
        with open(book_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = []
        book_name = Path(book_path).stem
        
        # Handle different JSON structures
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Extract text from various fields
                    text = ""
                    if 'text' in item:
                        text = item['text']
                    elif 'content' in item:
                        text = item['content']
                    elif 'instruction' in item and 'output' in item:
                        text = f"Question: {item['instruction']}\nAnswer: {item['output']}"
                    
                    if text and len(text.strip()) > 50:  # Min length filter
                        entries.append({
                            'text': text,
                            'source': book_name,
                            'metadata': item
                        })
        elif isinstance(data, dict):
            # Handle nested structures
            for key, value in data.items():
                if isinstance(value, str) and len(value.strip()) > 50:
                    entries.append({
                        'text': value,
                        'source': book_name,
                        'metadata': {'key': key}
                    })
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'text' in item:
                            entries.append({
                                'text': item['text'],
                                'source': book_name,
                                'metadata': item
                            })
        
        return entries
    
    def build_from_books(self, books_dir: str, chunk_size: int = 512):
        """
        Build vector database from all books in directory
        
        Args:
            books_dir: Directory containing book JSON files
            chunk_size: Size of text chunks
        """
        print(f"Building vector database from: {books_dir}")
        print(f"Using model: {self.model_name}")
        
        books_path = Path(books_dir)
        book_files = list(books_path.glob("*.json"))
        
        # Filter out metadata/stats files
        book_files = [f for f in book_files 
                     if 'stats' not in f.stem.lower() 
                     and 'metadata' not in f.stem.lower()
                     and 'combined' not in f.stem.lower()]
        
        print(f"Found {len(book_files)} books")
        
        all_chunks = []
        all_metadata = []
        
        for book_file in book_files:
            print(f"\nProcessing: {book_file.name}")
            
            try:
                entries = self.load_book_data(str(book_file))
                print(f"  Loaded {len(entries)} entries")
                
                chunk_count = 0
                for entry in entries:
                    chunks = self.chunk_text(entry['text'], chunk_size=chunk_size)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadata.append({
                            'source': entry['source'],
                            'chunk_id': f"{entry['source']}_{len(all_chunks)}",
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'original_metadata': entry.get('metadata', {})
                        })
                        chunk_count += 1
                
                print(f"  Created {chunk_count} chunks")
                
            except Exception as e:
                print(f"  ⚠️  Error processing {book_file.name}: {e}")
                continue
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print(f"\nTotal chunks: {len(self.chunks)}")
        print("Generating embeddings...")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        
        print(f"✓ Generated {len(self.embeddings)} embeddings")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
    
    def save(self, output_path: str):
        """Save vector database to disk"""
        data = {
            'model_name': self.model_name,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Calculate size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✓ Saved vector database to: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Chunks: {len(self.chunks)}")
        print(f"  Embeddings: {len(self.embeddings)}")
    
    @classmethod
    def load(cls, input_path: str):
        """Load vector database from disk"""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance without loading model initially
        instance = cls.__new__(cls)
        instance.model_name = data['model_name']
        instance.chunks = data['chunks']
        instance.embeddings = data['embeddings']
        instance.metadata = data['metadata']
        instance.model = None  # Load model on first search
        
        print(f"✓ Loaded vector database from: {input_path}")
        print(f"  Chunks: {len(instance.chunks)}")
        print(f"  Embedding dimension: {data['embedding_dim']}")
        
        return instance
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with text, score, and metadata
        """
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'score': float(similarities[idx]),
                'metadata': self.metadata[idx]
            })
        
        return results


def main():
    """Build vector database from books"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build vector database from Vedic astrology books')
    parser.add_argument('--books-dir', default='../data/raw/books', 
                       help='Directory containing book JSON files')
    parser.add_argument('--output', default='vector_db.pkl',
                       help='Output file for vector database')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                       help='Sentence transformer model')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Text chunk size in characters')
    
    args = parser.parse_args()
    
    # Build database
    db = VectorDatabase(model_name=args.model)
    db.build_from_books(args.books_dir, chunk_size=args.chunk_size)
    db.save(args.output)
    
    # Test search
    print("\n" + "="*60)
    print("Testing search functionality")
    print("="*60)
    
    test_queries = [
        "What is the significance of Jupiter?",
        "How to calculate Shadbala?",
        "What are Nakshatras?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = db.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (score: {result['score']:.3f}):")
            print(f"  Source: {result['metadata']['source']}")
            print(f"  Text: {result['text'][:200]}...")


if __name__ == '__main__':
    main()