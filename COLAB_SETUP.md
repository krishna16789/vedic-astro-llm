# Google Colab Training Setup Guide

Train your Vedic Astrology LLM on Google Colab's free GPU (much faster than MacBook Pro!)

## Quick Start

### Step 1: Prepare Your Files

**Option A: Push to GitHub (Recommended)**
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Prepare for Colab training"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/vedic-astro-llm.git
git push -u origin main
```

**Option B: Create ZIP File**
```bash
# On your Mac, create a ZIP of your project
cd ~/Desktop
zip -r vedic-astro-llm.zip vedic-astro-llm/ -x "*.git*" "*/venv/*" "*/__pycache__/*"
```

### Step 2: Open Google Colab

1. Go to https://colab.research.google.com/
2. Sign in with your Google account
3. Click **File → Upload notebook**
4. Upload the `colab_training.ipynb` file from your project

### Step 3: Enable GPU

1. In Colab, click **Runtime → Change runtime type**
2. Select **T4 GPU** (or L4 GPU if available)
3. Click **Save**

### Step 4: Run the Training

1. Run each cell in order (click the ▶️ button or press Shift+Enter)
2. When prompted, upload your ZIP file (if using Option B)
3. Training will start automatically
4. Monitor progress in real-time

## Training Time

- **Free T4 GPU**: ~3-4 hours
- **L4 GPU**: ~2-3 hours
- **A100 GPU** (Colab Pro): ~1-2 hours

vs. MacBook Pro: **24-48 hours** ⚠️

## Important Tips

### 1. Keep Colab Active
- Colab disconnects after ~90 minutes of inactivity
- Click in the notebook every hour to prevent disconnection
- Or install Colab Auto-Clicker extension

### 2. Download Checkpoints
Every 500 steps, a checkpoint is saved. Download them periodically:
```python
from google.colab import files
import shutil

# Download latest checkpoint
shutil.make_archive('checkpoint', 'zip', './training/checkpoints/colab-run/checkpoint-500')
files.download('checkpoint.zip')
```

### 3. Monitor Training
```python
# In a new cell, monitor training progress
!tail -f ./training/logs/events.out.tfevents.*
```

### 4. If Disconnected
- Checkpoints are saved every 500 steps
- Re-run cells to resume from last checkpoint
- Training will continue from where it left off

## After Training

### Download Your Model
The notebook automatically downloads the model after training. You'll get a `vedic-astro-model.zip` file.

### Use Locally
1. Extract the ZIP file on your Mac
2. Load and use:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load your trained adapters
model = PeftModel.from_pretrained(model, "./path/to/extracted/model")

# Generate
prompt = "Explain the 7th house in Vedic astrology:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### GPU Not Available
```
RuntimeError: No GPU found
```
**Solution**: Runtime → Change runtime type → Select T4 GPU

### Out of Memory
```
OutOfMemoryError: CUDA out of memory
```
**Solutions**:
- Reduce `per_device_train_batch_size` to 2
- Reduce `max_seq_length` to 1024
- Enable `gradient_checkpointing: true`

### Disconnection During Training
**Solution**: 
- Checkpoints are saved automatically
- Re-run cells to resume
- Consider Colab Pro ($10/month) for longer sessions

### Upload Timeout
**Solution**:
- Use GitHub instead (faster)
- Or split ZIP into smaller files

## Cost Comparison

| Option | Cost | Time | Quality |
|--------|------|------|---------|
| Google Colab Free | $0 | 3-4 hrs | Good |
| Colab Pro | $10/month | 2-3 hrs | Better |
| MacBook Pro M1/M2 | $0 | 24-48 hrs | Good |
| AWS/GCP GPU | $2-5 | 1-2 hrs | Best |

**Recommendation**: Start with free Colab, upgrade to Pro if you need faster training.

## Next Steps

After training:
1. Test your model with various prompts
2. Deploy as an API (FastAPI + Hugging Face)
3. Fine-tune further with domain-specific data
4. Share your model on Hugging Face Hub

## Support

If you encounter issues:
1. Check the error message in Colab
2. Verify GPU is enabled (Runtime → View resources)
3. Check CUDA memory usage
4. Restart runtime if needed