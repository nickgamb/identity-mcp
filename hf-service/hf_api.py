#!/usr/bin/env python3
"""
HuggingFace Inference Service with Tool Support
Provides OpenAI-compatible API for HuggingFace models with proper tool calling support
Supports GLM, GPT-OSS, and other HF models that support function calling
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import logging
import time
import threading
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HuggingFace Inference Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
# Set via HF_MODEL environment variable (default: THUDM/glm-4-5-air)
# Can be changed to any HuggingFace model that supports tools
MODEL_NAME = os.getenv("HF_MODEL", "THUDM/glm-4-5-air")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep-alive timeout (5 minutes default, like Ollama)
# Set HF_KEEP_ALIVE environment variable to change (0 = immediate, -1 = never, or seconds)
KEEP_ALIVE_SECONDS = int(os.getenv("HF_KEEP_ALIVE", "300"))  # 5 minutes default

# Global model and tokenizer
model = None
tokenizer = None
last_used = None
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

def load_model():
    """Load HuggingFace model and tokenizer from cache or download (lazy loading)"""
    global model, tokenizer, last_used
    
    with model_lock:
        if model is not None:
            last_used = datetime.now()
            return
        
        logger.info(f"Loading HuggingFace model {MODEL_NAME} on {DEVICE}...")
        
        # Check if MODEL_NAME is a local path
        is_local_path = os.path.isdir(MODEL_NAME) or os.path.exists(MODEL_NAME)
        
        if is_local_path:
            logger.info(f"Loading from local path: {MODEL_NAME}")
            local_files_only = True
        else:
            logger.info("Loading from HuggingFace (will check cache first, download if needed)")
            local_files_only = False
        
        # Use the mounted cache directory
        cache_dir = "/root/.cache/huggingface"
        
        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only
            )
            logger.info("Loading model (this may take a while if downloading)...")
            
            # Multi-GPU configuration
            num_gpus = torch.cuda.device_count() if DEVICE == "cuda" else 0
            logger.info(f"Detected {num_gpus} GPU(s)")
            
            if DEVICE == "cuda" and num_gpus > 1:
                # Multi-GPU: Use device_map="auto" to split model across GPUs
                # This will automatically distribute layers across available GPUs
                logger.info(f"Using multi-GPU mode: splitting model across {num_gpus} GPUs")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",  # Automatically splits across all GPUs
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    low_cpu_mem_usage=True,
                    max_memory={i: "22GiB" for i in range(num_gpus)}  # Limit per GPU to avoid OOM
                )
                # Log which GPUs are being used
                if hasattr(model, 'hf_device_map'):
                    logger.info(f"Model device map: {model.hf_device_map}")
            elif DEVICE == "cuda":
                # Single GPU
                logger.info("Using single GPU mode")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=cache_dir,
                    local_files_only=local_files_only
                )
            else:
                # CPU mode
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=None,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only
                )
                model = model.to(DEVICE)
            
            last_used = datetime.now()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if is_local_path:
                logger.error(f"Local model path {MODEL_NAME} not found or invalid")
            else:
                logger.info("If the model isn't in cache, it will download from HuggingFace")
            raise

def unload_model():
    """Unload model from GPU memory to free VRAM"""
    global model, tokenizer, last_used
    
    with model_lock:
        if model is None:
            return
        
        logger.info("Unloading model from GPU memory to free VRAM...")
        
        # Move model to CPU and delete
        if DEVICE == "cuda":
            model = model.cpu()
            torch.cuda.empty_cache()
        
        del model
        del tokenizer
        model = None
        tokenizer = None
        last_used = None
        
        logger.info("Model unloaded successfully")

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
    model_loaded = model is not None
    return {
        "status": "healthy", 
        "model": MODEL_NAME, 
        "device": DEVICE,
        "model_loaded": model_loaded,
        "keep_alive_seconds": KEEP_ALIVE_SECONDS
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    global model, tokenizer, last_used
    
    # Lazy load model on first request
    if model is None:
        load_model()
    else:
        last_used = datetime.now()
    
    try:
        # Detect model type for formatting
        model_type = "glm" if "glm" in MODEL_NAME.lower() else "default"
        
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
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME.split("/")[-1],
                "object": "model",
                "created": 1000000000,
                "owned_by": "glm-service"
            }
        ]
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

