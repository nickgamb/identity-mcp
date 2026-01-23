#!/usr/bin/env python3
"""
HuggingFace Inference Service with Dynamic Model Hot-Swapping
Provides OpenAI-compatible API for HuggingFace models with proper tool calling support

Dynamic Model Loading:
- Supports multiple base models (GLM, GPT-OSS, etc.)
- Hot-swaps between models on demand (unloads current, loads requested)
- Supports LoRA adapters for fine-tuned models
- Auto-unloads after idle timeout to free GPU memory
"""

import os
# Set CUDA memory allocation to reduce fragmentation (must be before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch

# Patch for PyTorch < 2.4: Add dummy torch.xpu module for transformers MXFP4 quantizer
# The MXFP4 quantizer checks torch.xpu.is_available() which doesn't exist in older PyTorch
if not hasattr(torch, 'xpu'):
    class DummyXPU:
        @staticmethod
        def is_available():
            return False
    torch.xpu = DummyXPU()

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging
import time
import threading
import gc
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _force_model_dtype_inplace(model, target_dtype):
    """Force a consistent dtype across the entire sharded model.
    This avoids float != bf16/fp16 mixed errors during inference."""
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)
        for b in model.buffers():
            if b.is_floating_point() and b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)
    return model

app = FastAPI(title="HuggingFace Inference Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base paths for models and adapters (mounted directories)
MODELS_PATH = os.getenv("HF_MODELS_PATH", "/app/models")
ADAPTERS_PATH = os.getenv("HF_ADAPTERS_PATH", "/app/adapters")

# Model registry: maps model names to HuggingFace model IDs and optional adapters
# Use local paths when available (faster, no downloads)
# To add a new model, just add an entry here - no docker-compose changes needed
MODEL_REGISTRY = {
    # GPT-OSS models - use local HF cache path
    "gpt-oss-20b": {
        "hf_model": f"{MODELS_PATH}/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
        "adapter": None
    },
    "gpt-oss-20b-finetuned": {
        "hf_model": f"{MODELS_PATH}/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
        "adapter": f"{ADAPTERS_PATH}/lora-gpt-oss-20b-1766515801769"
    },
    # GLM models
    "glm-4.5-air": {
        "hf_model": f"{MODELS_PATH}/glm-4.5-air",
        "adapter": None
    },
    # Add more models here as needed:
    # "model-name": {
    #     "hf_model": f"{MODELS_PATH}/model-folder",
    #     "adapter": f"{ADAPTERS_PATH}/adapter-folder" or None
    # },
}

# Keep-alive timeout (5 minutes default, like Ollama)
KEEP_ALIVE_SECONDS = int(os.getenv("HF_KEEP_ALIVE", "300"))

# Global model state
model = None
tokenizer = None
last_used = None
current_model_name = None  # Tracks which model is currently loaded
model_lock = threading.Lock()

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

def load_model(model_name: str):
    """Load HuggingFace model and tokenizer, with hot-swapping support.
    
    Args:
        model_name: Name of the model to load (must be in MODEL_REGISTRY)
    """
    global model, tokenizer, last_used, current_model_name
    
    # Look up model in registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_config = MODEL_REGISTRY[model_name]
    hf_model_path = model_config["hf_model"]
    adapter_path = model_config.get("adapter")
    
    with model_lock:
        # Check if requested model is already loaded
        if model is not None and current_model_name == model_name:
            last_used = datetime.now()
            return
        
        # Need to switch models - unload first if something is loaded
        if model is not None:
            logger.info(f"Hot-swapping from '{current_model_name}' to '{model_name}'...")
            _unload_model_internal()
        
        logger.info(f"Loading model '{model_name}' ({hf_model_path}) on {DEVICE}...")
        if adapter_path:
            logger.info(f"Will apply LoRA adapter from: {adapter_path}")
        
        # Check if model path is local
        is_local_path = os.path.isdir(hf_model_path) or os.path.exists(hf_model_path)
        
        if is_local_path:
            logger.info(f"Loading from local path: {hf_model_path}")
            local_files_only = True
        else:
            logger.info("Loading from HuggingFace (will check cache first, download if needed)")
            local_files_only = False
        
        cache_dir = "/root/.cache/huggingface"
        
        try:
            logger.info("Loading tokenizer...")
            # If adapter has its own tokenizer, use that; otherwise use base model's
            tokenizer_path = adapter_path if (adapter_path and os.path.exists(os.path.join(adapter_path, "tokenizer.json"))) else hf_model_path
            
            # Try fast tokenizer first, fall back to slow if it fails
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, 
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=is_local_path if tokenizer_path == hf_model_path else True
                )
            except Exception as e:
                logger.warning(f"Fast tokenizer failed ({e}), trying slow tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, 
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=is_local_path if tokenizer_path == hf_model_path else True,
                    use_fast=False
                )
            logger.info("Loading model (this may take a while if downloading)...")
            
            # Multi-GPU configuration with CPU offload support
            num_gpus = torch.cuda.device_count() if DEVICE == "cuda" else 0
            logger.info(f"Detected {num_gpus} GPU(s)")
            
            if DEVICE == "cuda" and num_gpus > 1:
                logger.info(f"Using multi-GPU mode: loading on CPU first then dispatching to {num_gpus} GPUs")
                # Load to CPU first to avoid GPU OOM during MXFP4 dequantization
                # Use bfloat16 for better precision (matches fine-tuning script)
                logger.info("Step 1: Loading/dequantizing model on CPU (uses RAM, avoids GPU OOM)...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,  # bfloat16 for better precision
                    device_map="cpu",  # Load entirely on CPU first
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=True,
                )
                # Force consistent dtype across all parameters
                base_model = _force_model_dtype_inplace(base_model, torch.bfloat16)
                
                logger.info("Step 2: Dispatching model to GPUs...")
                from accelerate import dispatch_model, infer_auto_device_map
                max_mem = {i: "20GiB" for i in range(num_gpus)}
                max_mem["cpu"] = "100GiB"  # Server has 115GB RAM
                
                # Use model's no_split_modules if available (prevents splitting transformer layers)
                no_split = getattr(base_model, "_no_split_modules", None)
                device_map = infer_auto_device_map(
                    base_model, 
                    max_memory=max_mem,
                    no_split_module_classes=no_split
                )
                logger.info(f"Computed device map: {device_map}")
                base_model = dispatch_model(base_model, device_map=device_map)
                
                # Clean up CPU memory after dispatch
                gc.collect()
                
                if hasattr(base_model, 'hf_device_map'):
                    logger.info(f"Model device map: {base_model.hf_device_map}")
            elif DEVICE == "cuda":
                logger.info("Using single GPU mode: loading on CPU first then dispatching")
                # Load to CPU first to avoid GPU OOM during dequantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=True,
                )
                base_model = _force_model_dtype_inplace(base_model, torch.bfloat16)
                
                logger.info("Dispatching model to GPU...")
                from accelerate import dispatch_model, infer_auto_device_map
                num_gpus_single = torch.cuda.device_count()
                max_mem_single = {i: "20GiB" for i in range(num_gpus_single)}
                max_mem_single["cpu"] = "100GiB"
                no_split = getattr(base_model, "_no_split_modules", None)
                device_map = infer_auto_device_map(
                    base_model,
                    max_memory=max_mem_single,
                    no_split_module_classes=no_split
                )
                base_model = dispatch_model(base_model, device_map=device_map)
                gc.collect()
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=None,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only
                )
                base_model = base_model.to(DEVICE)
            
            # Apply LoRA adapter if configured
            if adapter_path:
                logger.info(f"Applying LoRA adapter from: {adapter_path}")
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(base_model, adapter_path)
                    logger.info("LoRA adapter applied successfully!")
                except ImportError:
                    logger.error("PEFT library not installed. Install with: pip install peft")
                    raise
                except Exception as e:
                    logger.error(f"Failed to apply LoRA adapter: {e}")
                    raise
            else:
                model = base_model
            
            current_model_name = model_name
            last_used = datetime.now()
            logger.info(f"Model '{model_name}' loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

def _unload_model_internal():
    """Internal unload without lock - caller must hold model_lock"""
    global model, tokenizer, last_used, current_model_name
    
    if model is None:
        return
    
    logger.info(f"Unloading model '{current_model_name}' from GPU memory...")
    
    if DEVICE == "cuda":
        model = model.cpu()
        torch.cuda.empty_cache()
    
    del model
    del tokenizer
    model = None
    tokenizer = None
    last_used = None
    current_model_name = None
    
    logger.info("Model unloaded successfully")

def unload_model():
    """Unload model from GPU memory to free VRAM"""
    with model_lock:
        _unload_model_internal()

def check_and_unload_if_idle():
    """Check if model should be unloaded based on keep-alive timeout"""
    global last_used
    
    if KEEP_ALIVE_SECONDS == -1:  # Never unload
        return
    
    if KEEP_ALIVE_SECONDS == 0:  # Immediate unload after use
        # This will be handled in the request handler
        return
    
    with model_lock:
        if model is None or last_used is None:
            return
        
        idle_time = (datetime.now() - last_used).total_seconds()
        if idle_time >= KEEP_ALIVE_SECONDS:
            logger.info(f"Model idle for {idle_time:.0f}s (timeout: {KEEP_ALIVE_SECONDS}s), unloading...")
            unload_model()

def format_tools_for_model(tools: List[Dict[str, Any]], model_type: str = "glm") -> str:
    """Format tools in the model's expected format"""
    if not tools:
        return ""
    
    tool_descriptions = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        
        tool_descriptions.append(f"{name}: {description}")
        if parameters:
            tool_descriptions.append(f"  Parameters: {json.dumps(parameters, indent=2)}")
    
    return "\n".join(tool_descriptions)

def format_messages_for_model(messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]] = None, model_type: str = "glm") -> str:
    """Format messages in the model's chat format with tool support"""
    formatted = []
    
    # Add system message with tools if provided
    if tools:
        tool_text = format_tools_for_model(tools, model_type)
        formatted.append(f"System: You have access to the following tools:\n{tool_text}\n")
    
    for msg in messages:
        role = msg.role
        content = msg.content or ""
        
        if role == "system":
            formatted.append(f"System: {content}")
        elif role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            if msg.tool_calls:
                # Format tool calls in GLM's XML format
                for tool_call in msg.tool_calls:
                    func = tool_call.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                    formatted.append(f"Assistant: <tool_call><invoke name=\"{name}\">{args}</invoke></tool_call>")
            else:
                formatted.append(f"Assistant: {content}")
        elif role == "tool":
            # Format tool response
            formatted.append(f"Tool: {content}")
    
    return "\n".join(formatted)

def parse_model_response(text: str, tools: Optional[List[Dict[str, Any]]] = None, model_type: str = "glm") -> Dict[str, Any]:
    """Parse model response and extract tool calls if present"""
    import re
    
    # Check for tool calls in GLM format
    tool_call_pattern = r'<tool_call><invoke name="([^"]+)"[^>]*>([^<]+)</invoke></tool_call>'
    matches = re.findall(tool_call_pattern, text)
    
    if matches:
        tool_calls = []
        for name, args_str in matches:
            try:
                args = json.loads(args_str)
            except:
                args = args_str
            
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else args_str
                }
            })
        
        return {
            "content": None,
            "tool_calls": tool_calls
        }
    
    # No tool calls, return text content
    # Remove any tool call tags from response
    clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
    
    return {
        "content": clean_text,
        "tool_calls": None
    }

@app.get("/health")
async def health():
    """Health check endpoint - doesn't load model"""
    return {
        "status": "healthy",
        "available_models": list(MODEL_REGISTRY.keys()),
        "current_model": current_model_name,
        "model_loaded": model is not None,
        "device": DEVICE,
        "keep_alive_seconds": KEEP_ALIVE_SECONDS
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    global model, tokenizer, last_used
    
    # Normalize model name for lookup
    requested_model = request.model.lower().replace(":", "-").replace(".", "-")
    
    # Find matching model in registry
    model_name = None
    for name in MODEL_REGISTRY.keys():
        if name.lower() in requested_model or requested_model in name.lower():
            model_name = name
            break
    
    if not model_name:
        # Default to first available model
        model_name = list(MODEL_REGISTRY.keys())[0]
        logger.warning(f"Unknown model '{request.model}', defaulting to '{model_name}'")
    
    # Load appropriate model (handles hot-swapping if needed)
    load_model(model_name)
    
    try:
        # Detect model type for formatting
        model_type = "glm" if "glm" in model_name.lower() else "default"
        
        # Format messages for the model
        prompt = format_messages_for_model(request.messages, request.tools, model_type)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or 2048,
                temperature=request.temperature or 0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse response for tool calls
        parsed = parse_model_response(generated_text, request.tools, model_type)
        
        # Format response
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": parsed["content"]
            },
            "finish_reason": "stop"
        }
        
        if parsed["tool_calls"]:
            choice["message"]["tool_calls"] = parsed["tool_calls"]
            choice["message"]["content"] = None
            choice["finish_reason"] = "tool_calls"
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{hash(prompt)}",
            created=int(torch.randint(1000000000, 9999999999, (1,)).item()),
            model=request.model,
            choices=[choice],
            usage={
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                "total_tokens": outputs.shape[1]
            }
        )
        
        # Update last used time
        last_used = datetime.now()
        
        # If keep-alive is 0, unload immediately after use
        if KEEP_ALIVE_SECONDS == 0:
            # Schedule unload after response (don't block)
            threading.Thread(target=unload_model, daemon=True).start()
        
        return response.dict()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models from registry"""
    models = [
        {
            "id": name,
            "object": "model",
            "created": 1000000000,
            "owned_by": "hf-service"
        }
        for name in MODEL_REGISTRY.keys()
    ]
    return {
        "object": "list",
        "data": models
    }

def background_unload_checker():
    """Background thread to periodically check and unload idle models"""
    while True:
        time.sleep(60)  # Check every minute
        try:
            check_and_unload_if_idle()
        except Exception as e:
            logger.error(f"Error in background unload checker: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background unload checker thread"""
    if KEEP_ALIVE_SECONDS > 0:  # Only if not immediate unload
        thread = threading.Thread(target=background_unload_checker, daemon=True)
        thread.start()
        logger.info(f"Started background model unload checker (keep-alive: {KEEP_ALIVE_SECONDS}s)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

