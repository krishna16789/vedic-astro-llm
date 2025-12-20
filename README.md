# Vedic Astrology LLM - Fine-tuning Project

## Overview
This project creates a highly accurate fine-tuned Large Language Model for Vedic astrology using VedAstro's calculation libraries and extensive training datasets from classical texts.

## Project Goals
- Leverage VedAstro's astronomical calculation engine
- Fine-tune a base LLM with extensive Vedic astrology knowledge
- Incorporate B.V. Raman's books and classical texts
- Create a production-ready model without RAG implementation

## Architecture

```
vedic-astro-llm/
├── data/                    # Training datasets
│   ├── raw/                # Raw data from VedAstro
│   ├── processed/          # Cleaned and formatted data
│   └── final/              # Final training/validation splits
├── calculations/           # Core Vedic astrology calculations
│   ├── core/              # Python port of VedAstro calculations
│   ├── ephemeris/         # Swiss Ephemeris integration
│   └── utils/             # Helper functions
├── training/              # Model fine-tuning
│   ├── configs/          # Training configurations
│   ├── scripts/          # Training scripts
│   └── checkpoints/      # Model checkpoints
├── evaluation/           # Model evaluation
│   └── metrics/         # Custom metrics
├── inference/           # Inference engine
│   └── api/            # REST API
└── docs/               # Documentation
```

## Data Sources

### From VedAstro
1. **Horoscope Predictions** - B.V. Raman's classical interpretations (3000+ samples)
2. **ML Datasets** - 100 years of astronomical data (London 1900-2000)
3. **Event Data** - Astrological event calculations and XML definitions
4. **Marriage & Life Events** - 15K famous people dataset
5. **Planetary Data** - Comprehensive planetary position datasets

### Additional Sources (to be added)
- Brihat Parashara Hora Shastra
- Jataka Parijata
- Phaladeepika
- Saravali
- Modern commentaries and interpretations

## Technology Stack

- **Base Model**: LLaMA 2 or Mistral (7B/13B parameters)
- **Fine-tuning**: QLoRA/LoRA for efficient training
- **Framework**: PyTorch + Transformers (HuggingFace)
- **Calculations**: Python + Swiss Ephemeris
- **Data Processing**: Pandas, NumPy
- **Training**: Accelerate, PEFT, bitsandbytes
- **Deployment**: FastAPI + Docker

## Key Features

1. **Astronomical Calculations**: Direct integration with Swiss Ephemeris for accurate planetary positions
2. **Classical Knowledge**: Training data from B.V. Raman and traditional texts
3. **Structured Output**: Generates coherent astrological interpretations
4. **Multi-dimensional Analysis**: Houses, planets, signs, aspects, dasas, yogas
5. **No RAG Required**: All knowledge embedded in model weights

## Setup

### Prerequisites
```bash
# Python 3.10+
# CUDA-capable GPU (16GB+ VRAM recommended)
# 50GB+ disk space
```

### Installation
```bash
# Clone repository
git clone <repo-url>
cd vedic-astro-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Swiss Ephemeris
pip install pyswisseph
```

## Usage

### Data Preparation
```bash
# Extract and process VedAstro datasets
python scripts/prepare_data.py --source /path/to/VedAstro

# Generate training samples
python scripts/generate_training_data.py --output data/processed/
```

### Training
```bash
# Fine-tune base model
python training/scripts/train.py \
  --config training/configs/llama2-7b-lora.yaml \
  --data data/final/ \
  --output training/checkpoints/

# Monitor with TensorBoard
tensorboard --logdir training/logs/
```

### Inference
```bash
# Run inference API
python inference/api/server.py --model training/checkpoints/final/

# Example request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "birth_time": "1990-01-01T10:30:00",
    "location": {"lat": 28.6139, "lon": 77.2090},
    "query": "What are the career prospects?"
  }'
```

## Training Data Format

The model is trained on structured instruction-following format:

```json
{
  "instruction": "House1LordInHouse1",
  "input": "Birth details and context",
  "output": "Astrological interpretation based on classical texts"
}
```

## Model Capabilities

The fine-tuned model can:
- Interpret planetary positions and houses
- Analyze yogas (planetary combinations)
- Predict dasas and bhuktis
- Provide marriage compatibility analysis
- Offer career and life event predictions
- Explain astrological concepts

## Evaluation Metrics

1. **Accuracy**: Comparison with expert astrologers
2. **Consistency**: Same inputs produce same outputs
3. **Classical Adherence**: Alignment with traditional texts
4. **Calculation Precision**: Astronomical accuracy

## Roadmap

- [x] Extract VedAstro calculation libraries
- [x] Process training datasets
- [ ] Port C# calculations to Python
- [ ] Create training pipeline
- [ ] Fine-tune base LLM
- [ ] Build inference API
- [ ] Add more classical texts
- [ ] Create evaluation framework
- [ ] Deploy production model

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## License

MIT License - See LICENSE file

## Acknowledgments

- VedAstro.org for open-source calculation libraries
- B.V. Raman for classical astrological texts
- Swiss Ephemeris for astronomical calculations
- HuggingFace for transformer models and tools

## References

1. B.V. Raman - "Hindu Predictive Astrology"
2. B.V. Raman - "Three Hundred Important Combinations"
3. VedAstro Calculation Library - https://github.com/VedAstro/VedAstro
4. Swiss Ephemeris - https://www.astro.com/swisseph/

## Contact

For questions and support, please open an issue on GitHub.