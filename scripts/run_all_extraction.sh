#!/bin/bash
# Master script to extract and prepare all training data

set -e  # Exit on error

echo "=================================================="
echo "Vedic Astrology LLM - Data Extraction Pipeline"
echo "=================================================="

# Create necessary directories
echo -e "\n[1/5] Creating directories..."
mkdir -p data/raw/vedastro
mkdir -p data/raw/books
mkdir -p data/final

# Extract VedAstro data
echo -e "\n[2/5] Extracting VedAstro datasets..."
python scripts/extract_vedastro_data.py \
    --vedastro-dir ~/Downloads/VedAstro \
    --output-dir ./data/raw/vedastro

# Extract book data
echo -e "\n[3/5] Extracting astrology books..."
python scripts/extract_book_data.py \
    --books-dir "$HOME/Downloads/astrology books" \
    --output-dir ./data/raw/books

# Prepare training data
echo -e "\n[4/5] Preparing final training datasets..."
python scripts/prepare_training_data.py \
    --raw-data-dir ./data/raw \
    --output-dir ./data/final

# Summary
echo -e "\n[5/5] Data extraction complete!"
echo "=================================================="
echo "Training data is ready at: ./data/final/"
echo "  - train.json: Training dataset"
echo "  - validation.json: Validation dataset"
echo "  - metadata.json: Dataset statistics"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review the data in data/final/"
echo "  2. Start training with: python training/scripts/train.py --config training/configs/llama2-7b-lora.yaml"
echo ""