# RAG (Retrieval Augmented Generation) System for Vedic Astrology

## 🎯 Overview

This RAG system enhances your fine-tuned Vedic Astrology LLM with:

1. **Vector Database**: Semantic search across all Vedic astrology books
2. **Tool Calling**: Direct access to chart calculations (Shadbala, Bhavabala, Dasha, etc.)
3. **Context Enhancement**: Relevant knowledge retrieval for every question

## 📚 Components

### 1. Vector Database (`build_vector_db.py`)
- Processes all books in `data/raw/books/`
- Creates embeddings using Sentence Transformers
- Stores vectors in `vector_db.pkl` for quick loading
- Enables semantic search (meaning-based, not just keywords)

### 2. Tool Registry (`tools.py`)
- 8 calculation tools for chart analysis
- Direct access to:
  - Current chart data
  - Shadbala calculations
  - Bhavabala calculations
  - Dasha periods
  - Nakshatra attributes
  - Planet details
  - House details
  - Divisional charts

### 3. RAG Handler (`rag_handler.py`)
- Combines vector search + tool calling
- Enhances prompts with retrieved context
- Parses and executes tool calls from LLM
- Formats results back to LLM

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r rag/requirements.txt
```

This installs:
- `sentence-transformers` - For embeddings
- `numpy` - For vector operations
- `torch` - For model inference

### Step 2: Build Vector Database
```bash
python rag/build_vector_db.py \
  --books-dir data/raw/books \
  --output rag/vector_db.pkl \
  --model all-MiniLM-L6-v2 \
  --chunk-size 512
```

**Options:**
- `--model`: Choose embedding model
  - `all-MiniLM-L6-v2` (default) - Fast, 80MB, good quality
  - `all-mpnet-base-v2` - Better quality, 420MB, slower
- `--chunk-size`: Text chunk size (default: 512 characters)

**Expected Output:**
```
Building vector database from: data/raw/books
Using model: all-MiniLM-L6-v2
Found 11 books

Processing: Brihat-Jataka_text.json
  Loaded 145 entries
  Created 892 chunks

...

Total chunks: 8,543
Generating embeddings...
✓ Generated 8,543 embeddings
  Embedding dimension: 384

✓ Saved vector database to: rag/vector_db.pkl
  Size: 12.45 MB
  Chunks: 8,543
  Embeddings: 8,543
```

### Step 3: Start Server with RAG
```bash
cd server
python app.py
```

The server will automatically:
- Detect `rag/vector_db.pkl`
- Load vector database
- Enable RAG and tool calling
- Show: `✓ RAG system initialized with vector database`

## 🎨 How It Works

### Query Flow

```
User Question
     ↓
1. Vector Search → Retrieve 3 most relevant text chunks from books
     ↓
2. Tool Detection → Check if calculations needed (Shadbala, Dasha, etc.)
     ↓
3. Enhanced Prompt = Question + Retrieved Context + Tool Definitions
     ↓
4. Fine-Tuned Model → Generate response (may include tool calls)
     ↓
5. Tool Execution → Execute any requested calculations
     ↓
6. Final Response = LLM Answer + Calculation Results
```

### Example 1: Question with RAG

**User:** "What is the significance of Jupiter?"

**RAG Process:**
1. Search vector DB for "Jupiter significance"
2. Retrieve 3 relevant chunks from BPHS, Brihat Jataka, etc.
3. Add chunks to prompt as context
4. Model generates answer using both training AND retrieved knowledge

**Response includes:**
- Training data knowledge (fine-tuned model)
- Retrieved text from classical sources
- Citations to source books

### Example 2: Question with Tools

**User:** "What is the Shadbala of Jupiter in this chart?"

**Tool Process:**
1. Model recognizes need for calculation
2. Generates: `<tool>calculate_shadbala{"planet": "Jupiter"}</tool>`
3. RAG handler executes tool with current chart
4. Returns Shadbala scores
5. Model incorporates results in answer

**Response includes:**
- Shadbala scores (Sthana Bala, Dig Bala, etc.)
- Total strength
- Interpretation based on scores

### Example 3: Complex Query

**User:** "Analyze the strength of my 10th house and its lord"

**Combined Process:**
1. Vector search: "10th house strength lord"
2. Retrieve knowledge about 10th house significance
3. Model identifies needs:
   - House details tool
   - Bhavabala tool
   - Shadbala for house lord
4. Execute all three tools
5. Combine retrieved knowledge + calculations
6. Generate comprehensive analysis

## 🛠️ Available Tools

### 1. get_current_chart
Get current birth chart data with all planetary positions.

**Usage:**
```
<tool>get_current_chart{}</tool>
```

**Returns:**
- Ascendant details
- All planetary positions
- Nakshatras
- Signs

### 2. calculate_shadbala
Calculate six-fold strength for planet(s).

**Usage:**
```
<tool>calculate_shadbala{"planet": "Jupiter"}</tool>
<tool>calculate_shadbala{"planet": "all"}</tool>
```

**Parameters:**
- `planet` (optional): "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", or "all"

**Returns:**
- Sthana Bala (Positional Strength)
- Dig Bala (Directional Strength)
- Kaala Bala (Temporal Strength)
- Chesta Bala (Motional Strength)
- Naisargika Bala (Natural Strength)
- Drik Bala (Aspectual Strength)
- Total Shadbala
- Required minimum strength
- Strength percentage

### 3. calculate_bhavabala
Calculate house strength for house(s).

**Usage:**
```
<tool>calculate_bhavabala{"house": 10}</tool>
<tool>calculate_bhavabala{}</tool>
```

**Parameters:**
- `house` (optional): 1-12, or omit for all houses

**Returns:**
- Bhava Digbala
- Bhava Drishti Bala
- Total Bhavabala
- Strength rating

### 4. get_dasha_periods
Get Vimshottari Dasha timeline.

**Usage:**
```
<tool>get_dasha_periods{"years_ahead": 20}</tool>
```

**Parameters:**
- `years_ahead` (optional): Years to calculate (default: 10)

**Returns:**
- Current Mahadasha
- Current Antardasha
- Current Pratyantardasha
- Timeline of future periods

### 5. get_nakshatra_attributes
Get comprehensive nakshatra attributes.

**Usage:**
```
<tool>get_nakshatra_attributes{}</tool>
```

**Returns:**
- Atmakaraka
- Amatyakaraka
- Other Charakarakas
- Yoni (animal symbol)
- Gana (temperament)
- Tara (birth star group)

### 6. get_planet_details
Get detailed planet information.

**Usage:**
```
<tool>get_planet_details{"planet": "Saturn"}</tool>
```

**Parameters:**
- `planet` (required): Planet name

**Returns:**
- Position (sign, degree, nakshatra)
- Sign lord
- Nakshatra lord
- Speed
- Retrograde status
- Combustion status

### 7. get_house_details
Get house information including planets and lordship.

**Usage:**
```
<tool>get_house_details{"house": 7}</tool>
```

**Parameters:**
- `house` (required): House number (1-12)

**Returns:**
- House sign
- Sign lord
- Planets in house
- Their degrees

### 8. calculate_divisional_chart
Get positions in divisional charts.

**Usage:**
```
<tool>calculate_divisional_chart{"varga": "D9"}</tool>
```

**Parameters:**
- `varga` (required): "D1", "D9", "D10", "D12", "D16", "D20", "D24", "D27", "D30", "D60"

**Returns:**
- Ascendant in varga
- All planetary positions in varga

## 📊 Vector Database Details

### Embedding Model
Default: `all-MiniLM-L6-v2`
- Dimension: 384
- Size: ~80MB
- Speed: Fast (thousands of queries/sec)
- Quality: Good for most use cases

Alternative: `all-mpnet-base-v2`
- Dimension: 768
- Size: ~420MB
- Speed: Medium
- Quality: Better semantic understanding

### Chunking Strategy
- Chunk size: 512 characters (configurable)
- Overlap: 50 characters
- Smart sentence boundary detection
- Preserves context at chunk boundaries

### Search Method
- Cosine similarity
- Returns top-k most relevant chunks
- Includes source book metadata
- Relevance scores (0.0 - 1.0)

## 🧪 Testing

### Test Vector Database
```bash
# After building vector DB
python rag/build_vector_db.py --books-dir data/raw/books --output rag/vector_db.pkl

# Will automatically test searches:
# - "What is the significance of Jupiter?"
# - "How to calculate Shadbala?"
# - "What are Nakshatras?"
```

### Test Tool Calling
```python
from rag.tools import get_tool_registry
from calculations.core.ephemeris import VedicEphemeris
from datetime import datetime

# Create test chart
ephemeris = VedicEphemeris()
chart = ephemeris.get_chart_data(
    datetime(1990, 1, 1, 10, 30),
    28.6139, 77.2090
)

# Get tool registry
tools = get_tool_registry()

# Test Shadbala calculation
result = tools.execute_tool(
    "calculate_shadbala",
    {"planet": "Jupiter"},
    {"chart_data": chart, "datetime": "1990-01-01T10:30:00"}
)

print(result)
```

### Test RAG Handler
```python
from rag.rag_handler import get_rag_handler

# Initialize handler
rag = get_rag_handler("rag/vector_db.pkl")

# Test context retrieval
results = rag.retrieve_context("What is Jupiter?", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Source: {r['metadata']['source']}")
    print(f"Text: {r['text'][:200]}...\n")
```

## 🎯 API Response Format

With RAG and tools enabled, API responses include:

```json
{
  "response": "Jupiter is the most benefic planet...",
  "model": "mistral-7b-vedic-astrology-finetuned",
  "device": "mps",
  "fine_tuned": true,
  "rag_enabled": true,
  "tools_enabled": true,
  "tool_calls": [
    {
      "tool": "calculate_shadbala",
      "arguments": {"planet": "Jupiter"},
      "result": {
        "success": true,
        "result": {
          "sthana_bala": 245.3,
          "total_shadbala": 423.7,
          ...
        }
      }
    }
  ]
}
```

## 📈 Performance

### Memory Usage
- Vector DB: ~12-50 MB (depends on book count)
- Embedding Model: ~80-420 MB
- Total Additional: ~100-500 MB

### Speed
- Vector search: <10ms
- Tool execution: 10-100ms (depends on calculation)
- Total overhead: Usually <200ms

### Quality Improvement
- **Without RAG**: Model relies only on training data
- **With RAG**: Model accesses all source texts semantically
- **With Tools**: Model gets exact calculations on demand

## 🔧 Configuration

### In server/app.py

RAG is automatically enabled if:
1. `rag/vector_db.pkl` exists
2. `sentence-transformers` is installed

To disable:
```python
rag_handler = None  # Set this in app.py
```

### Build Options

**Fast build (smaller file):**
```bash
python rag/build_vector_db.py \
  --books-dir data/raw/books \
  --output rag/vector_db.pkl \
  --model all-MiniLM-L6-v2 \
  --chunk-size 512
```

**Quality build (larger file):**
```bash
python rag/build_vector_db.py \
  --books-dir data/raw/books \
  --output rag/vector_db.pkl \
  --model all-mpnet-base-v2 \
  --chunk-size 768
```

## 🐛 Troubleshooting

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### "Vector database not found"
```bash
python rag/build_vector_db.py --books-dir data/raw/books --output rag/vector_db.pkl
```

### "Tool execution failed"
- Ensure chart data is calculated first
- Check tool arguments match schema
- Verify calculations modules are working

### "RAG not improving responses"
- Check vector DB was built successfully
- Verify books contain relevant content
- Try different chunk sizes
- Use better embedding model (mpnet)

## 🌟 Benefits

### 1. Knowledge Access
- Access ALL source texts, not just training data
- Find relevant information semantically
- Get direct quotes from classical texts

### 2. Accurate Calculations
- No hallucinated numbers
- Exact Shadbala/Bhavabala scores
- Precise Dasha timings

### 3. Combined Intelligence
- Fine-tuned model understanding
- Retrieved classical knowledge
- Live calculation results
- = Comprehensive, accurate answers

## 🎉 Summary

Your Vedic Astrology LLM now has:
- ✅ Fine-tuned on 4,395 examples
- ✅ Access to all source books via RAG
- ✅ 8 calculation tools
- ✅ Semantic search capability
- ✅ Tool calling framework

**Result**: Most comprehensive Vedic Astrology AI assistant! 🔮✨