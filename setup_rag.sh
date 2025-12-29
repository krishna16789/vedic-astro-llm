#!/bin/bash

# Setup RAG System for Vedic Astrology LLM
# This script installs dependencies and builds the vector database

echo "=========================================="
echo "Setting up RAG System"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "📦 Step 1: Installing dependencies..."
pip install sentence-transformers numpy torch

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"

# Step 2: Build vector database
echo ""
echo "🔨 Step 2: Building vector database..."
echo "This will process all books in data/raw/books/ and create embeddings"
echo "Expected time: 2-5 minutes"
echo ""

python rag/build_vector_db.py \
    --books-dir data/raw/books \
    --output rag/vector_db.pkl \
    --model all-MiniLM-L6-v2 \
    --chunk-size 512

# Check if build was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to build vector database"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ RAG System Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start the server: cd server && python app.py"
echo "2. Open browser: http://localhost:5001"
echo "3. Try the Chat with AI tab"
echo ""
echo "Features enabled:"
echo "  ✓ RAG: Semantic search across all books"
echo "  ✓ Tools: 8 calculation tools available"
echo "  ✓ Fine-tuned model: Vedic astrology trained"
echo ""
echo "Happy Charting! 🔮✨"