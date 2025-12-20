# Getting Started with Vedic Astrology LLM

This guide will help you set up and train your own Vedic Astrology LLM from scratch.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 100GB+ free disk space

### Software Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher
- **Git**: For cloning repositories

## Installation

### 1. Clone the Repository

```bash
cd ~/Desktop
# The project is already created at vedic-astro-llm/
cd vedic-astro-llm
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Swiss Ephemeris Data (Optional but Recommended)

**On macOS/Linux (using curl):**

```bash
# Download ephemeris data files
mkdir -p ~/.swisseph
cd ~/.swisseph

# Download asteroid ephemeris
curl -O https://www.astro.com/ftp/swisseph/ephe/seas_18.se1

# Download Moon ephemeris
curl -O https://www.astro.com/ftp/swisseph/ephe/semo_18.se1

# Download planet ephemeris
curl -O https://www.astro.com/ftp/swisseph/ephe/sepl_18.se1
```

**Alternative: Let pyswisseph download automatically**

The pyswisseph library can download ephemeris data automatically when needed. If you skip this step, the library will download the necessary files on first use.

**Manual download (if needed):**

You can also download the files manually from:
https://www.astro.com/ftp/swisseph/ephe/

And place them in `~/.swisseph/` directory.

## Data Extraction

### Quick Start - Extract All Data

Run the master extraction script:

```bash
chmod +x scripts/run_all_extraction.sh
./scripts/run_all_extraction.sh
```

This will:
1. Extract VedAstro datasets (horoscope predictions, yogas, events)
2. Extract text from astrology books (avoiding OCR)
3. Prepare final training datasets with train/validation splits

### Manual Extraction (Optional)

If you prefer step-by-step extraction:

#### Step 1: Extract VedAstro Data

```bash
python scripts/extract_vedastro_data.py \
    --vedastro-dir ~/Downloads/VedAstro \
    --output-dir ./data/raw/vedastro
```

#### Step 2: Extract Book Data

```bash
python scripts/extract_book_data.py \
    --books-dir "$HOME/Downloads/astrology books" \
    --output-dir ./data/raw/books
```

#### Step 3: Prepare Training Data

```bash
python scripts/prepare_training_data.py \
    --raw-data-dir ./data/raw \
    --output-dir ./data/final
```

## Verify Data

Check that data was extracted successfully:

```bash
# View statistics
cat data/final/metadata.json

# Check sample count
wc -l data/final/train.json
wc -l data/final/validation.json

# View a sample
head -n 50 data/final/train.json
```

## Model Training

### Configure Training (Optional)

Edit the training configuration if needed:

```bash
nano training/configs/llama2-7b-lora.yaml
```

Key parameters to adjust:
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per GPU (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `lora_r`: LoRA rank (default: 16)

### Start Training

```bash
python training/scripts/train.py \
    --config training/configs/llama2-7b-lora.yaml
```

### Monitor Training

#### Using TensorBoard

```bash
# In a new terminal
tensorboard --logdir training/logs
```

Then open http://localhost:6006 in your browser.

#### Using Weights & Biases (Optional)

1. Sign up at https://wandb.ai
2. Login: `wandb login`
3. Training metrics will be automatically logged

### Training Time

Estimated training time on different GPUs:
- **RTX 4090**: ~6-8 hours for 3 epochs
- **A100**: ~4-5 hours for 3 epochs
- **RTX 3090**: ~8-10 hours for 3 epochs

## Testing the Model

### Quick Test

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and adapter
base_model = "meta-llama/Llama-2-7b-hf"
adapter_path = "./training/checkpoints/llama2-7b-lora/final"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# Test inference
prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain the Gajakesari Yoga in Vedic astrology

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Using the Calculation Library

### Example: Calculate Birth Chart

```python
from datetime import datetime
from calculations.core.ephemeris import VedicEphemeris

# Initialize
eph = VedicEphemeris()

# Calculate chart
birth_time = datetime(1990, 1, 1, 10, 30, 0)
latitude = 28.6139  # New Delhi
longitude = 77.2090

chart = eph.get_chart_data(birth_time, latitude, longitude)

# Print results
print(f"Ascendant: {chart['ascendant']['sign']}")
for planet, data in chart['planets'].items():
    print(f"{planet}: {data['sign']} ({data['nakshatra']})")
```

## Common Issues

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce batch size** in config:
   ```yaml
   per_device_train_batch_size: 2  # Reduce from 4
   gradient_accumulation_steps: 8  # Increase from 4
   ```

2. **Use smaller sequence length**:
   ```yaml
   max_seq_length: 1024  # Reduce from 2048
   ```

3. **Enable CPU offloading**:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       device_map="auto",
       offload_folder="offload",
       offload_state_dict=True
   )
   ```

### Swiss Ephemeris Errors

If you get ephemeris errors:

```bash
# Install from source
pip install --no-cache-dir pyswisseph --force-reinstall
```

### PDF Extraction Issues

If book extraction fails:

1. Check if PDFs are searchable (not scanned images)
2. Install additional dependencies:
   ```bash
   pip install PyPDF2 pdfplumber
   ```

## Next Steps

1. **Improve the model**: Add more training data from additional books
2. **Fine-tune further**: Experiment with different hyperparameters
3. **Deploy**: Create an API server for inference
4. **Evaluate**: Test model accuracy against expert predictions

## Resources

- **VedAstro Documentation**: https://vedastro.org/Docs
- **Swiss Ephemeris**: https://www.astro.com/swisseph/
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **LoRA Paper**: https://arxiv.org/abs/2106.09685

## Support

For questions and issues:
- Check existing documentation
- Review the code comments
- Open an issue on GitHub

## License

This project is licensed under the MIT License. See LICENSE file for details.