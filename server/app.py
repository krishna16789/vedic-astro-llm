"""
Vedic Astrology LLM Server
Flask API server for model interaction and calculations
Updated with divisional chart support
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from calculations.core.ephemeris import VedicEphemeris
from calculations.core.shadbala_enhanced import EnhancedShadbalaCalculator
from calculations.core.bhavabala_enhanced import EnhancedBhavabalaCalculator
from calculations.core.dasha import VimshottariDasha
from calculations.core.nakshatra_attributes import NakshatraAttributes
from calculations.core.prediction_generator import PredictionGenerator
from calculations.core.astro_tools import AstroTools

# Try to import RAG components
try:
    from rag.rag_handler import get_rag_handler
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️  RAG system not available. Run: python rag/build_vector_db.py --books-dir data/raw/books --output rag/vector_db.pkl")

# Try to import transformers for local model loading
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Chat will use mock responses.")

app = Flask(__name__)
CORS(app)

# Initialize calculators
ephemeris = VedicEphemeris()
shadbala_calc = EnhancedShadbalaCalculator()
bhavabala_calc = EnhancedBhavabalaCalculator()
dasha_calc = VimshottariDasha()
nakshatra_calc = NakshatraAttributes()
prediction_gen = PredictionGenerator()
astro_tools = AstroTools()

# Initialize RAG handler if available
rag_handler = None
if RAG_AVAILABLE:
    try:
        vector_db_path = Path(__file__).parent.parent / "rag" / "vector_db.pkl"
        if vector_db_path.exists():
            rag_handler = get_rag_handler(str(vector_db_path))
            print("✓ RAG system initialized with vector database")
        else:
            print("⚠️  Vector database not found. Build it with: python rag/build_vector_db.py")
            rag_handler = get_rag_handler(None)  # No vector DB, tools only
    except Exception as e:
        print(f"⚠️  Could not initialize RAG system: {e}")
        rag_handler = None

# Initialize local Mistral model
model = None
tokenizer = None

# Store last calculated chart globally (simple session storage)
last_calculated_chart = None
last_chart_timestamp = None

# Detect best available device (Apple Silicon MPS, NVIDIA CUDA, or CPU)
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Detected Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Detected NVIDIA GPU (CUDA)")
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

model_loaded = False

# Configuration options - adjust these based on your system
LOAD_MODEL_ON_STARTUP = False  # Set to False to prevent crashes - load on first request instead
USE_FINE_TUNED_MODEL = False   # TEMPORARILY DISABLED - Testing base Mistral on MPS
# Use absolute path to checkpoint (works regardless of where server is run from)
CHECKPOINT_PATH = str(Path(__file__).parent.parent / "training" / "checkpoints" / "mistral-7b-lora" / "final")
USE_4BIT_QUANTIZATION = True   # Reduces memory from ~14GB to ~4GB
USE_8BIT_QUANTIZATION = False  # Alternative: ~7GB memory
MAX_MEMORY_GB = 8              # Maximum memory to use (adjust based on your system)

def load_local_model():
    """Load local Mistral 7B model with memory-efficient settings"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        print("✓ Model already loaded")
        return True
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  Transformers not available. Install with: pip install transformers torch")
        return False
    
    try:
        # Check if we should use fine-tuned model
        if USE_FINE_TUNED_MODEL and os.path.exists(CHECKPOINT_PATH):
            model_name = "mistralai/Mistral-7B-v0.1"
            print(f"\n{'=' * 60}")
            print(f"🎯 Loading FINE-TUNED MODEL (FINAL CHECKPOINT)")
            print(f"Path: {CHECKPOINT_PATH}")
            print(f"Base model: {model_name}")
            print(f"Training: COMPLETE - Vedic Astrology Trained!")
            print(f"Memory settings:")
            print(f"  - 4-bit quantization: {USE_4BIT_QUANTIZATION}")
            print(f"  - 8-bit quantization: {USE_8BIT_QUANTIZATION}")
            print(f"  - Max memory: {MAX_MEMORY_GB}GB")
            print(f"{'=' * 60}\n")
        else:
            model_name = "mistralai/Mistral-7B-v0.1"
            print(f"\n{'=' * 60}")
            print(f"⚠️  Loading BASE MODEL (no astrology training)")
            print(f"Loading {model_name} on {device}...")
            print(f"Memory settings:")
            print(f"  - 4-bit quantization: {USE_4BIT_QUANTIZATION}")
            print(f"  - 8-bit quantization: {USE_8BIT_QUANTIZATION}")
            print(f"  - Max memory: {MAX_MEMORY_GB}GB")
            print(f"{'=' * 60}\n")
        
        # Load tokenizer (minimal memory)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Tokenizer loaded")
        
        # Prepare model loading arguments with memory optimizations
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        # Add quantization if requested (requires bitsandbytes - only works on CUDA)
        if USE_4BIT_QUANTIZATION and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                print("Using 4-bit quantization (saves ~10GB memory)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("⚠️  bitsandbytes not installed, quantization disabled")
                print("   Install with: pip install bitsandbytes")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        elif USE_8BIT_QUANTIZATION and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                print("Using 8-bit quantization (saves ~7GB memory)")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("⚠️  bitsandbytes not installed, quantization disabled")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: f"{MAX_MEMORY_GB}GB"}
        elif device == "mps":
            # Apple Silicon MPS - use float16 for efficiency
            # Note: bitsandbytes quantization is not supported on MPS
            print("✓ Using Apple Silicon MPS backend")
            print("⚠️  Note: Quantization not supported on MPS, using float16")
            model_kwargs["torch_dtype"] = torch.float16
            # MPS doesn't support device_map="auto", load to MPS manually after loading
        else:
            # CPU mode - use float32 but with memory constraints
            model_kwargs["torch_dtype"] = torch.float32
            print(f"⚠️  Loading on CPU - this will be slow!")
            print(f"   Consider using GPU or reducing MAX_MEMORY_GB")
        
        # Load model
        print("Loading base model... (this may take 2-5 minutes)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Load LoRA adapters if using fine-tuned model
        if USE_FINE_TUNED_MODEL and os.path.exists(CHECKPOINT_PATH):
            try:
                from peft import PeftModel
                print(f"\n🔧 Loading LoRA adapters from checkpoint...")
                model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
                print(f"✓ LoRA adapters loaded successfully!")
                print(f"✓ Model is now FINE-TUNED for Vedic Astrology!")
            except ImportError:
                print("⚠️  peft not installed. Install with: pip install peft")
                print("⚠️  Using base model without fine-tuning")
            except Exception as e:
                print(f"⚠️  Error loading LoRA adapters: {e}")
                print("⚠️  Using base model without fine-tuning")
        
        # Move to MPS if that's our device (MPS doesn't support device_map="auto")
        if device == "mps":
            model = model.to(device)
        
        model.eval()  # Set to evaluation mode
        
        if USE_FINE_TUNED_MODEL and os.path.exists(CHECKPOINT_PATH):
            print(f"\n✓✓✓ FINE-TUNED VEDIC ASTROLOGY MODEL LOADED SUCCESSFULLY! ✓✓✓")
            print(f"✓ Running on: {device}")
            print(f"✓ Model is now ready to answer Vedic astrology questions!")
        else:
            print(f"✓ Base model loaded successfully on {device}")
        
        # Print memory stats
        if device == "cuda":
            print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"✓ GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        elif device == "mps":
            print(f"✓ Model loaded on Apple Silicon GPU")
            print(f"⚠️  MPS memory stats not available via PyTorch")
        
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Falling back to mock mode")
        print("\n💡 Tips to reduce memory usage:")
        print("   1. Set USE_4BIT_QUANTIZATION = True in app.py")
        print("   2. Reduce MAX_MEMORY_GB")
        print("   3. Close other applications")
        print("   4. Use the Chart Calculator (always works!)\n")
        return False

# Conditionally load model on startup
if LOAD_MODEL_ON_STARTUP:
    print("\n" + "=" * 60)
    print("Initializing Mistral 7B Model on Startup...")
    print("=" * 60)
    model_loaded = load_local_model()
else:
    print("\n" + "=" * 60)
    print("Model Loading: DEFERRED")
    print("=" * 60)
    print("Model will load on first chat request (prevents crashes)")
    print("Use the Chart Calculator immediately!")
    print("=" * 60 + "\n")


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'ephemeris_available': True
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with local Mistral model
    Request: {"message": "user message", "history": [...], "chart_data": {...}}
    """
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])
    chart_data = data.get('chart_data', None)  # Optional chart context
    
    # If no chart_data provided, use last calculated chart
    if chart_data is None and last_calculated_chart is not None:
        chart_data = last_calculated_chart
        print(f"✓ Using last calculated chart from {last_chart_timestamp}")
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Load model on first request if not loaded yet
    if not model_loaded and TRANSFORMERS_AVAILABLE:
        print("\n🔄 Loading model on first request...")
        load_local_model()
    
    # If local model is loaded, use it
    if model is not None and tokenizer is not None:
        try:
            # Use simplified format (RAG handler format is too detailed and overwhelms model)
            # Disable RAG handler for now - use simple direct format
            if False and rag_handler is not None:  # Temporarily disabled - ALWAYS use house format
                enhanced_prompt, context = rag_handler.build_enhanced_prompt(
                    user_message=user_message,
                    chart_data=chart_data,
                    history=history,
                    use_rag=False,
                    use_tools=False
                )
                conversation = f"<s>[INST] {enhanced_prompt} [/INST]"
                print(f"🔍 Enhanced prompt length: {len(enhanced_prompt)} chars")
            else:
                # VedAstro-style: Generate weighted predictions + tool calling
                system_prompt = """You are a confident expert Vedic astrologer with access to precise calculation tools.

You have access to the following tools to get exact astrological calculations:
- calculate_planet_strength: Get Shadbala strength of a planet
- calculate_house_strength: Get Bhavabala strength of a house
- get_navamsa_position: Get D9 Navamsa position of a planet
- check_planetary_aspect: Check if planets aspect each other (includes Rahu/Ketu, mutual aspects)
- check_retrograde_status: Check if a planet is retrograde
- check_combustion: Check if a planet is combust
- get_current_dasha: Get current Vimshottari Dasha
- get_planet_dignity: Get complete dignity (exaltation, debilitation, own sign, moolatrikona, etc.)
- check_planetary_friendship: Check natural friendship between planets

When you need precise calculations, use these tools by calling:
TOOL_CALL: <tool_name>(<arguments>)

Example: TOOL_CALL: calculate_planet_strength({"planet_name": "Jupiter"})

Based on the PREDICTIONS provided (with strength weights) and any tool results, answer the user's question.
Judge predictions by their Weight - higher weight means stronger influence.
Provide a direct, confident answer based on the most relevant predictions and calculations."""
                
                # Generate predictions with weights (VedAstro architecture)
                chart_context = ""
                tool_definitions = ""
                if chart_data:
                    try:
                        # Format actual chart data for LLM instead of trying to regenerate
                        chart_context = "\n\n" + "="*60 + "\n"
                        chart_context += "BIRTH CHART ANALYSIS\n"
                        chart_context += "="*60 + "\n\n"
                        
                        # Ascendant
                        if 'ascendant' in chart_data:
                            asc = chart_data['ascendant']
                            chart_context += f"ASCENDANT (Lagna):\n"
                            chart_context += f"  • Sign: {asc.get('sign', 'N/A')}\n"
                            chart_context += f"  • Degree: {asc.get('degree_in_sign', 0):.2f}°\n"
                            chart_context += f"  • Nakshatra: {asc.get('nakshatra', 'N/A')} (Pada {asc.get('nakshatra_pada', 'N/A')})\n\n"
                        
                        # Planets with house positions
                        if 'planets' in chart_data:
                            chart_context += "PLANETARY POSITIONS:\n"
                            for planet, data in chart_data['planets'].items():
                                chart_context += f"\n{planet}:\n"
                                chart_context += f"  • Sign: {data.get('sign', 'N/A')}\n"
                                chart_context += f"  • House: {data.get('house', 'N/A')}\n"
                                chart_context += f"  • Degree: {data.get('degree_in_sign', 0):.2f}°\n"
                                chart_context += f"  • Nakshatra: {data.get('nakshatra', 'N/A')} (Pada {data.get('nakshatra_pada', 'N/A')})\n"
                                if data.get('is_retrograde'):
                                    chart_context += f"  • Status: RETROGRADE ®\n"
                                if data.get('is_combust'):
                                    chart_context += f"  • Status: COMBUST (weakened by Sun)\n"
                        
                        chart_context += "\n" + "="*60 + "\n"
                        chart_context += "IMPORTANT:\n"
                        chart_context += "• Use the tools to calculate precise strengths and dignities\n"
                        chart_context += "• Analyze house placements for life areas\n"
                        chart_context += "• Check aspects between planets\n"
                        chart_context += "• Provide comprehensive predictions based on calculations\n"
                        chart_context += "="*60 + "\n"
                        
                        # Add tool availability notice
                        tools_list = astro_tools.get_tool_definitions()
                        tool_definitions = "\n\nAVAILABLE CALCULATION TOOLS:\n"
                        tool_definitions += "="*60 + "\n"
                        for tool in tools_list:
                            func = tool["function"]
                            tool_definitions += f"• {func['name']}: {func['description']}\n"
                        tool_definitions += "="*60 + "\n"
                        
                    except Exception as e:
                        print(f"⚠️ Chart formatting failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to simple message
                        chart_context = "\n\nChart data available but formatting failed.\n"
                        tool_definitions = ""
                
                # Simple conversation format - CRITICAL: Add space after [/INST] to help model start
                if len(history) == 0:
                    conversation = f"<s>[INST] {system_prompt}{tool_definitions}{chart_context}\n\n{user_message} [/INST] "
                else:
                    # No history for now to keep it simple
                    conversation = f"<s>[INST] {system_prompt}{tool_definitions}{chart_context}\n\n{user_message} [/INST] "
                
                # Build context for tool execution
                # Get JD from chart metadata if available
                jd = 0
                birth_time = datetime.utcnow()
                if chart_data and 'meta' in chart_data:
                    # Try to reconstruct birth time from metadata
                    if 'datetime' in chart_data['meta']:
                        try:
                            birth_time = datetime.fromisoformat(chart_data['meta']['datetime'])
                            jd = ephemeris.get_julian_day(birth_time)
                        except:
                            pass
                
                context = {
                    "chart_data": chart_data,
                    "chart": chart_data if chart_data else {},
                    "jd": jd,
                    "birth_time": birth_time
                }
                
                print(f"🔍 Compact prompt length: {len(conversation)} chars")
                print(f"🔍 Chart data present: {chart_data is not None}")
            print(conversation)
            # Tokenize with MAXIMUM context window
            # Mistral 7B officially supports 8192 tokens - use full capacity since running locally
            max_context_length = 8192  # MAXIMUM for Mistral 7B
            inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=max_context_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"🔍 Input tokens: {inputs['input_ids'].shape[1]} / {max_context_length} max")
            
            # Generate LONG responses (2048 tokens = ~1500 words)
            max_tokens = 200 if device == "cpu" else 2048  # DOUBLED for comprehensive analysis
            min_tokens = 50 if device == "cpu" else 150  # Minimum length to prevent early stopping
            print(f"🔍 Generating {min_tokens}-{max_tokens} new tokens (approx {min_tokens * 0.75}-{max_tokens * 0.75} words)...")
            
            try:
                with torch.no_grad():
                    # Use safer generation parameters for MPS/float16 stability
                    # Avoid sampling which can cause NaN with float16
                    if device == "mps":
                        # Greedy decoding for MPS stability
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            min_new_tokens=min_tokens,
                            do_sample=False,  # Greedy = no sampling instability
                            repetition_penalty=1.15,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        # Standard sampling for CUDA
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            min_new_tokens=min_tokens,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=50,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            num_beams=1,
                        )
            except Exception as gen_error:
                print(f"❌ Generation error: {gen_error}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Model generation error: {str(gen_error)}',
                    'response': f"The model encountered an error during generation. This may be due to:\n1. Memory constraints on {device}\n2. Input length ({inputs['input_ids'].shape[1]} tokens)\n3. MPS backend compatibility issues\n\nTry:\n- Asking a shorter question\n- Restarting the server\n- Using the Chart Calculator instead",
                    'using_mock': True
                }), 500
            
            print(f"🔍 Generated tokens: {outputs.shape[1]}")
            print(f"🔍 New tokens generated: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
            
            try:
                # Move outputs to CPU and clamp token IDs to valid range
                outputs_cpu = outputs.cpu()
                vocab_size = tokenizer.vocab_size
                
                # Clamp token IDs to valid vocabulary range to prevent overflow
                outputs_cpu = torch.clamp(outputs_cpu, 0, vocab_size - 1)
                
                # Decode response with special tokens visible for debugging
                full_response_with_tokens = tokenizer.decode(outputs_cpu[0], skip_special_tokens=False)
                full_response = tokenizer.decode(outputs_cpu[0], skip_special_tokens=True)
                
                # Debug: Show the actual tokens generated
                new_token_ids = outputs_cpu[0][inputs['input_ids'].shape[1]:]
                print(f"🔍 First 10 new token IDs: {new_token_ids[:10].tolist()}")
                print(f"🔍 Vocab size: {vocab_size}")
                print(f"🔍 Max token ID in output: {outputs_cpu[0].max().item()}")
                print(f"🔍 Min token ID in output: {outputs_cpu[0].min().item()}")
                try:
                    print(f"🔍 First few new tokens: {tokenizer.decode(new_token_ids[:20], skip_special_tokens=False)}")
                except:
                    print(f"🔍 Could not decode first tokens (invalid IDs)")
                
                # Extract only the new response (after the last [/INST])
                assistant_message = full_response.split("[/INST]")[-1].strip()
                print(f"🔍 Extracted message length: {len(assistant_message)} chars")
                print(f"🔍 First 200 chars: {assistant_message[:200]}")
            except Exception as decode_error:
                print(f"❌ Decoding error: {decode_error}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Response decoding error: {str(decode_error)}',
                    'response': "Generated response but failed to decode it properly.",
                    'using_mock': True
                }), 500
            
            # Debug: Log the initial response
            print(f"🔍 Initial model response length: {len(assistant_message)} chars")
            if len(assistant_message) < 10:
                print(f"⚠️  Very short response: '{assistant_message}'")
                print(f"🔍 Full response with tokens: {full_response_with_tokens[-200:]}")  # Last 200 chars
            
            # Process tool calls from LLM response
            tool_results = []
            try:
                if chart_data and "TOOL_CALL:" in assistant_message:
                    # Extract and execute tool calls
                    import re
                    import json
                    tool_pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
                    matches = re.findall(tool_pattern, assistant_message)
                    
                    for tool_name, args_str in matches:
                        print(f"🔧 Executing tool: {tool_name}")
                        try:
                            arguments = json.loads(args_str) if args_str else {}
                            result = astro_tools.execute_tool(tool_name, arguments, context)
                            tool_results.append({
                                "tool": tool_name,
                                "arguments": arguments,
                                "result": result
                            })
                            print(f"✓ Tool result: {result}")
                        except Exception as e:
                            print(f"⚠️  Tool execution error for {tool_name}: {e}")
                            tool_results.append({
                                "tool": tool_name,
                                "error": str(e)
                            })
                    
                    # Remove tool calls from response and add results
                    assistant_message = re.sub(tool_pattern, '', assistant_message).strip()
                    
                    # Append tool results to response
                    if tool_results:
                        assistant_message += "\n\n📊 Calculation Results:\n"
                        for tr in tool_results:
                            if "error" not in tr:
                                assistant_message += f"\n• {tr['tool']}: {json.dumps(tr['result'], indent=2)}\n"
                    
                    print(f"✓ After tool processing: {len(assistant_message)} chars, {len(tool_results)} tools executed")
            except Exception as tool_error:
                print(f"⚠️  Tool processing error: {tool_error}")
                import traceback
                traceback.print_exc()
                # Keep original response if tool processing fails
            
            # Fallback if response is empty
            if not assistant_message or len(assistant_message.strip()) == 0:
                assistant_message = "I apologize, but I generated an empty response. Please try rephrasing your question or providing more context."
                print("⚠️  Empty response detected, using fallback message")
            
            model_info = 'mistral-7b-vedic-astrology-finetuned' if USE_FINE_TUNED_MODEL else 'mistral-7b-base'
            
            try:
                response_data = {
                    'response': assistant_message,
                    'model': model_info,
                    'device': device,
                    'fine_tuned': USE_FINE_TUNED_MODEL,
                    'rag_enabled': rag_handler is not None and rag_handler.vector_db is not None,
                    'tools_enabled': rag_handler is not None
                }
                
                # Add tool results if any
                if tool_results:
                    response_data['tool_calls'] = tool_results
                
                print(f"✓ Returning response with {len(assistant_message)} chars")
                return jsonify(response_data)
            except Exception as json_error:
                print(f"❌ JSON serialization error: {json_error}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Response serialization error: {str(json_error)}',
                    'response': str(assistant_message)[:500],  # Return truncated response
                    'using_mock': True
                }), 500
            
        except Exception as e:
            return jsonify({
                'error': f'Model error: {str(e)}',
                'using_mock': True,
                'response': f"I encountered an error processing your request. Please try again or use the Chart Calculator for planetary position calculations."
            }), 500
    
    else:
        # Mock response when model is not loaded
        # Provide helpful context about the chart if available
        if chart_data and 'ascendant' in chart_data:
            asc = chart_data['ascendant']
            planets_summary = []
            if 'planets' in chart_data:
                for planet, pos in list(chart_data['planets'].items())[:5]:  # First 5 planets
                    planets_summary.append(f"{planet} in {pos.get('sign', '')} ({pos.get('nakshatra', '')})")
            
            mock_response = f"""📊 CHART ANALYSIS (Mock Mode - Base Model Not Trained)

⚠️ IMPORTANT: The base Mistral model isn't trained for Vedic astrology!
You're seeing random responses because this model needs fine-tuning with your training data.

However, I can see your chart data:
• Ascendant: {asc.get('sign', '')} at {asc.get('degree_in_sign', 0):.2f}°
• Key Planets: {', '.join(planets_summary[:3])}

💡 TO GET REAL ASTROLOGY INSIGHTS:
1. Fine-tune the model using your training data in /data/final/
2. Or use the Chart Calculator for accurate calculations
3. Or use an API-based astrology model

You asked: "{user_message}"

The random response you got earlier is because the base model has no astrology knowledge."""
        else:
            mock_response = f"""⚠️ MODEL NOT READY FOR ASTROLOGY

The base Mistral 7B model isn't trained for Vedic astrology! You're getting random responses because:

1. This is a general language model (knows about bullets, not planets!)
2. It needs fine-tuning with your Vedic astrology training data
3. The training data is in /data/final/ but hasn't been used yet

💡 OPTIONS:
• Use the Chart Calculator (always accurate!)
• Fine-tune the model with your training data
• Wait for model loading (2-5 minutes on first request)
• Use an API-based astrology service

You asked: "{user_message}"

For now, use the Chart Calculator tab for real planetary calculations!"""
        
        return jsonify({
            'response': mock_response,
            'model': 'mock',
            'using_mock': True,
            'note': 'Base model not trained for astrology - use Chart Calculator or fine-tune model'
        })


@app.route('/api/calculate-chart', methods=['POST'])
def calculate_chart():
    """
    Calculate Vedic astrology chart
    Request: {
        "datetime": "2023-01-01T10:30:00",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timezone_offset": 5.5  // Optional: hours offset from UTC (e.g., IST = 5.5)
    }
    
    Note: Swiss Ephemeris requires UTC time. If timezone_offset is provided,
    the datetime will be converted from local time to UTC automatically.
    """
    data = request.json
    
    try:
        # Parse datetime
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset (hours from UTC, e.g., IST = 5.5)
        timezone_offset = data.get('timezone_offset', 0)
        
        # Convert local time to UTC
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        # Get coordinates
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        if lat == 0 and lon == 0:
            return jsonify({'error': 'Valid latitude and longitude required'}), 400
        
        # Calculate chart using UTC time
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Store chart globally for chat context
        global last_calculated_chart, last_chart_timestamp
        last_calculated_chart = chart
        last_chart_timestamp = dt_local.isoformat()
        
        print(f"✓ Chart calculated and stored for chat context")
        
        return jsonify({
            'success': True,
            'chart': chart,
            'timezone_info': {
                'local_time': dt_local.isoformat(),
                'utc_time': dt_utc.isoformat(),
                'offset_hours': timezone_offset
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500


@app.route('/api/planet-position', methods=['POST'])
def planet_position():
    """
    Get specific planet position
    Request: {
        "planet": "Sun",
        "datetime": "2023-01-01T10:30:00"
    }
    """
    data = request.json
    
    try:
        planet_name = data.get('planet')
        dt_str = data.get('datetime')
        
        if not planet_name or not dt_str:
            return jsonify({'error': 'planet and datetime are required'}), 400
        
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        jd = ephemeris.get_julian_day(dt)
        
        position = ephemeris.get_planet_position(planet_name, jd)
        
        return jsonify({
            'success': True,
            'planet': planet_name,
            'position': {
                'longitude': position.longitude,
                'latitude': position.latitude,
                'sign': position.sign,
                'degree_in_sign': position.degree_in_sign,
                'nakshatra': position.nakshatra,
                'pada': position.nakshatra_pada,
                'speed': position.speed
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Calculation error: {str(e)}'}), 500


@app.route('/api/available-planets', methods=['GET'])
def available_planets():
    """Get list of available planets"""
    return jsonify({
        'planets': list(ephemeris.PLANETS.keys())
    })


@app.route('/api/search-location', methods=['GET'])
def search_location():
    """
    Search for location coordinates by place name
    Uses Nominatim (OpenStreetMap) geocoding API
    """
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        import urllib.parse
        import urllib.request
        import json
        
        # Use Nominatim API (free, no API key needed)
        encoded_query = urllib.parse.quote(query)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=5"
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'VedicAstroLLM/1.0')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        
        # Format results
        locations = []
        for item in data:
            locations.append({
                'name': item.get('display_name', ''),
                'latitude': float(item.get('lat', 0)),
                'longitude': float(item.get('lon', 0)),
                'type': item.get('type', ''),
            })
        
        return jsonify({
            'success': True,
            'locations': locations
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Location search failed: {str(e)}',
            'locations': []
        }), 500


@app.route('/api/generate-rasi-chart', methods=['POST'])
def generate_rasi_chart():
    """
    Generate Rasi chart (D1) data for visualization
    South Indian style: Signs are in fixed positions, planets move through them
    """
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset and convert to UTC
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data using UTC time
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Get ascendant sign number (0-11) where 0=Aries, 1=Taurus, etc.
        ascendant_degree = chart['ascendant']['degree']
        ascendant_sign_num = int(ascendant_degree / 30)
        ascendant_sign = ephemeris.SIGNS[ascendant_sign_num]
        
        # Initialize all 12 zodiac signs (Aries=1 to Pisces=12)
        # In South Indian chart, signs are fixed positions
        signs = {}
        for i in range(12):
            sign_num = i  # 0=Aries, 1=Taurus, ... 11=Pisces
            signs[i+1] = {  # Sign numbers 1-12 for display
                'number': i + 1,
                'sign': ephemeris.SIGNS[sign_num],
                'sign_num': sign_num,
                'planets': [],
                'has_ascendant': sign_num == ascendant_sign_num
            }
        
        # Place each planet in its zodiac sign
        for planet_name, planet_data in chart['planets'].items():
            planet_sign_num = int(planet_data['longitude'] / 30)
            sign_key = planet_sign_num + 1  # Convert to 1-12
            
            signs[sign_key]['planets'].append({
                'name': planet_name,
                'degree': planet_data['degree_in_sign'],
                'longitude': planet_data['longitude'],
                'speed': planet_data['speed'],
                'is_retrograde': planet_data.get('is_retrograde', False),
                'is_combust': planet_data.get('is_combust', False)
            })
        
        return jsonify({
            'success': True,
            'chart_type': 'Rasi (D1) - South Indian',
            'ascendant': {
                'sign': ascendant_sign,
                'degree': chart['ascendant']['degree_in_sign'],
                'sign_num': ascendant_sign_num
            },
            'signs': signs  # Changed from 'houses' to 'signs'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Chart generation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate-divisional-chart', methods=['POST'])
def generate_divisional_chart():
    """
    Generate divisional chart (Varga) data for visualization
    Currently supports D1 (Rasi) and D9 (Navamsa)
    """
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        varga = data.get('varga', 'D9')
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        
        # Get timezone offset and convert to UTC
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get base chart data
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Calculate divisional chart positions
        ascendant_degree = chart['ascendant']['degree']
        divisional_ascendant = ephemeris.calculate_divisional_chart(ascendant_degree, varga)
        divisional_ascendant_sign_num = int(divisional_ascendant / 30)
        divisional_ascendant_sign = ephemeris.SIGNS[divisional_ascendant_sign_num]
        
        # Initialize all 12 zodiac signs
        signs = {}
        for i in range(12):
            sign_num = i
            signs[i+1] = {
                'number': i + 1,
                'sign': ephemeris.SIGNS[sign_num],
                'sign_num': sign_num,
                'planets': [],
                'has_ascendant': sign_num == divisional_ascendant_sign_num
            }
        
        # Place each planet in its divisional chart sign
        for planet_name, planet_data in chart['planets'].items():
            divisional_longitude = ephemeris.calculate_divisional_chart(planet_data['longitude'], varga)
            planet_sign_num = int(divisional_longitude / 30)
            sign_key = planet_sign_num + 1
            
            signs[sign_key]['planets'].append({
                'name': planet_name,
                'degree': divisional_longitude % 30,
                'longitude': divisional_longitude,
                'speed': planet_data['speed'],
                'is_retrograde': planet_data.get('is_retrograde', False),
                'is_combust': planet_data.get('is_combust', False)
            })
        
        return jsonify({
            'success': True,
            'chart_type': f'{varga} Chart',
            'ascendant': {
                'sign': divisional_ascendant_sign,
                'degree': divisional_ascendant % 30,
                'sign_num': divisional_ascendant_sign_num
            },
            'signs': signs
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Divisional chart generation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/calculate-shadbala', methods=['POST'])
def calculate_shadbala():
    """Calculate Shadbala (Six-fold strength) for all planets"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        jd = ephemeris.get_julian_day(dt_utc)
        
        # Calculate Shadbala for each planet (using enhanced calculator)
        shadbala_results = {}
        for planet_name, planet_data in chart['planets'].items():
            if planet_name not in ['Rahu', 'Ketu']:
                planet_with_name = {**planet_data, 'name': planet_name}
                shadbala = shadbala_calc.calculate_shadbala(planet_with_name, chart, jd, dt_local)
                shadbala_results[planet_name] = shadbala
        
        return jsonify({
            'success': True,
            'shadbala': shadbala_results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Shadbala calculation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/calculate-bhavabala', methods=['POST'])
def calculate_bhavabala():
    """Calculate Bhavabala (House strength) for all 12 houses"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        jd = ephemeris.get_julian_day(dt_utc)
        
        # Calculate Shadbala first for more accurate Bhavabala (using enhanced calculator)
        shadbala_results = {}
        for planet_name, planet_data in chart['planets'].items():
            if planet_name not in ['Rahu', 'Ketu']:
                planet_with_name = {**planet_data, 'name': planet_name}
                shadbala = shadbala_calc.calculate_shadbala(planet_with_name, chart, jd, dt_local)
                shadbala_results[planet_name] = shadbala
        
        # Calculate Bhavabala for all houses
        bhavabala_results = bhavabala_calc.calculate_all_houses(chart, shadbala_results)
        
        return jsonify({
            'success': True,
            'bhavabala': bhavabala_results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Bhavabala calculation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/calculate-dasha', methods=['POST'])
def calculate_dasha():
    """Calculate Vimshottari Dasha periods (Mahadasha/Antardasha)"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data to find Moon's position
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        moon_longitude = chart['planets']['Moon']['longitude']
        
        # Get current dasha
        current_dasha = dasha_calc.get_current_dasha(dt_local, moon_longitude)
        
        # Get complete dasha timeline (entire 120-year cycle)
        timeline = dasha_calc.get_dasha_timeline(dt_local, moon_longitude, years_ahead=120)
        
        return jsonify({
            'success': True,
            'current_dasha': current_dasha,
            'timeline': timeline
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Dasha calculation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500



@app.route('/api/calculate-nakshatra-attributes', methods=['POST'])
def calculate_nakshatra_attributes():
    """Calculate comprehensive nakshatra attributes including Karakas, Yoni, Gana, Tara"""
    data = request.json
    
    try:
        dt_str = data.get('datetime')
        if not dt_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        dt_local = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        timezone_offset = data.get('timezone_offset', 0)
        from datetime import timedelta
        dt_utc = dt_local - timedelta(hours=timezone_offset)
        
        lat = float(data.get('latitude', 0))
        lon = float(data.get('longitude', 0))
        
        # Get chart data
        chart = ephemeris.get_chart_data(dt_utc, lat, lon)
        
        # Calculate all nakshatra attributes
        attributes = nakshatra_calc.get_chart_attributes(chart)
        
        return jsonify({
            'success': True,
            'attributes': attributes
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Nakshatra attributes calculation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Vedic Astrology LLM Server")
    print("=" * 60)
    print(f"✓ Ephemeris calculations: Ready")
    print(f"✓ Mistral 7B Model: {'Loaded on ' + device if model_loaded else 'Not loaded (mock mode)'}")
    print(f"✓ Device: {device}")
    print("\nStarting server on http://localhost:5001")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
