#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LoRA Fine-Tuning for Large Language Models                ║
║                                                                              ║
║  Supports models up to 20B+ parameters on consumer/prosumer hardware         ║
║  • CPU-only mode (slow but always works)                                     ║
║  • GPU+CPU offload mode (DeepSpeed ZeRO-3 with CPU offload)                  ║
║  • Multi-GPU support with automatic sharding                                 ║
║  • Checkpoint save/resume with full state preservation                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE EXAMPLES:

  CPU-ONLY (guaranteed to work, ~40GB RAM for 20B model):
    python finetune_lora.py --model_name gpt-oss:20b --cpu_only --epochs 3
    python finetune_lora.py --model_name gpt-oss:20b --max_length 2048 --cpu_only --epochs 3

  SINGLE GPU + CPU OFFLOAD (24GB VRAM + 100GB+ RAM):
    python finetune_lora.py --model_name gpt-oss:20b --epochs 3
    python finetune_lora.py --model_name gpt-oss:20b --max_length 2048 --epochs 3

  MULTI-GPU + CPU OFFLOAD (2x24GB VRAM + 100GB+ RAM):
    deepspeed --num_gpus=2 finetune_lora.py --model_name gpt-oss:20b --epochs 3
    deepspeed --num_gpus=2 finetune_lora.py --model_name gpt-oss:20b --max_length 2048 --epochs 3

  RESUME FROM CHECKPOINT:
    python finetune_lora.py --model_name gpt-oss:20b \\
        --output_name my-adapter-1234567890 \\
        --resume_from_checkpoint checkpoint-1500

  EXPORT DATASET ONLY:
    python finetune_lora.py --export_only --dataset_source all

MONITORING:
  watch -n 1 nvidia-smi                    # GPU utilization
  tail -f adapters/*/training_*.log        # Training progress
"""

import os
import sys
import json
import argparse
import logging
import random
import gc
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Training configuration with sensible defaults.
    
    Memory guidance for 20B models:
    - max_length=512:  ~20GB VRAM per GPU with offload
    - max_length=1024: ~30GB VRAM per GPU with offload  
    - max_length=2048: ~50GB+ VRAM, needs aggressive offload or CPU-only
    
    For 2x P40 (48GB total) with 2048 length: use CPU-only or reduce to 1024
    """
    model_name: str = "gpt-oss:20b"
    dataset_source: str = "all"  # conversations|files|memories|all
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 2048  # Default to longer sequences
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    checkpoint_steps: int = 500
    logging_steps: int = 50
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42

# Project paths - support multi-user via USER_ID env var
PROJECT_ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.name == "conversation_processing" else Path(__file__).parent
USER_ID = os.environ.get("USER_ID")

def get_user_dir(base_dir: Path) -> Path:
    """Get user-specific directory if USER_ID is set."""
    if USER_ID:
        return base_dir / USER_ID
    return base_dir

CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations")
FILES_DIR = get_user_dir(PROJECT_ROOT / "files")
MEMORY_DIR = get_user_dir(PROJECT_ROOT / "memory")
TRAINING_DATA_DIR = get_user_dir(PROJECT_ROOT / "training_data")
ADAPTERS_DIR = get_user_dir(PROJECT_ROOT / "adapters")

# Ensure directories exist
for d in [TRAINING_DATA_DIR, ADAPTERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    """Colored log output for better readability."""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        if record.levelname == 'INFO':
            # Clean info messages without prefix
            return f"{color}{record.getMessage()}{self.RESET}"
        return f"{color}{self.BOLD}[{record.levelname}]{self.RESET} {color}{record.getMessage()}{self.RESET}"


def setup_logging(output_dir: Optional[Path] = None, rank: int = 0) -> logging.Logger:
    """Setup logging to both console and file."""
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler with colors (only rank 0)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)
    
    # File handler (all ranks write to separate files)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank{rank}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
        if rank == 0:
            logger.info(f"📝 Log file: {log_file}")
    
    return logger


def print_banner(title: str, char: str = "═"):
    """Print a nice banner."""
    width = 70
    print(f"\n\033[1m{char * width}\033[0m")
    print(f"\033[1m  {title.center(width - 4)}\033[0m")
    print(f"\033[1m{char * width}\033[0m\n")


def print_config_table(config: dict):
    """Print configuration as a nice table."""
    max_key = max(len(str(k)) for k in config.keys())
    for key, value in config.items():
        print(f"  \033[36m{key:<{max_key}}\033[0m : \033[33m{value}\033[0m")
    print()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def get_gpu_info() -> dict:
    """Get GPU information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "devices": []}
        
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024**3)
            mem_free = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
            devices.append({
                "name": props.name,
                "memory_total_gb": round(mem_total, 1),
                "memory_free_gb": round(mem_free, 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "has_tensor_cores": props.major >= 7,  # Volta+ has tensor cores
            })
        
        return {
            "available": True,
            "count": len(devices),
            "devices": devices,
        }
    except Exception as e:
        return {"available": False, "count": 0, "devices": [], "error": str(e)}


def get_model_id(model_name: str) -> str:
    """Convert model name to HuggingFace model ID."""
    mapping = {
        'gpt-oss:20b': 'openai/gpt-oss-20b',
        'gpt-oss/20b': 'openai/gpt-oss-20b',
    }
    if model_name in mapping:
        return mapping[model_name]
    if ':' in model_name:
        parts = model_name.split(':')
        return f"{parts[0]}/{parts[0]}-{parts[1]}"
    return model_name


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_conversations() -> List[Dict[str, Any]]:
    """Load conversations from JSONL files."""
    conversations = []
    jsonl_files = sorted(CONVERSATIONS_DIR.glob('conversation_*.jsonl'))
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                messages = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if msg.get('role') and msg.get('content') is not None:
                            messages.append({
                                'role': msg.get('role'),
                                'content': msg.get('content', ''),
                            })
                    except json.JSONDecodeError:
                        continue
                
                if messages:
                    conversations.append({
                        'id': jsonl_file.stem.replace('conversation_', ''),
                        'messages': messages
                    })
        except Exception as e:
            logging.warning(f"Error loading {jsonl_file}: {e}")
    
    return conversations


def load_files() -> List[Dict[str, Any]]:
    """Load text files from files directory."""
    files = []
    if not FILES_DIR.exists():
        return files
    
    for ext in ['*.txt', '*.md']:
        for file_path in FILES_DIR.rglob(ext):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                title = file_path.stem
                if content.startswith('# '):
                    first_line = content.split('\n')[0]
                    title = first_line[2:].strip()
                
                files.append({
                    'filename': file_path.name,
                    'title': title,
                    'content': content
                })
            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")
    
    return files


def load_memory() -> List[Dict[str, Any]]:
    """Load memory records from JSONL files."""
    memory_records = []
    if not MEMORY_DIR.exists():
        return memory_records
    
    for jsonl_file in MEMORY_DIR.glob('*.jsonl'):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        memory_records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.warning(f"Error loading {jsonl_file}: {e}")
    
    return memory_records


def export_training_data(source: str = "all", output_path: Optional[Path] = None) -> Path:
    """Export training data to JSONL format."""
    if output_path is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        output_path = TRAINING_DATA_DIR / f"dataset-{timestamp}.jsonl"
    
    training_examples = []
    
    # Load conversations
    if source in ["conversations", "all"]:
        conversations = load_conversations()
        for conv in conversations:
            messages = conv.get('messages', [])
            for i in range(len(messages) - 1):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    training_examples.append({
                        'instruction': user_msg.get('content', ''),
                        'response': assistant_msg.get('content', '')
                    })
        print(f"  📚 Loaded {len(training_examples)} examples from conversations")
    
    # Load files
    if source in ["files", "all"]:
        start_count = len(training_examples)
        files = load_files()
        for file_data in files:
            title = file_data.get('title', file_data.get('filename', ''))
            training_examples.append({
                'instruction': f"Read this document: {title}",
                'response': file_data.get('content', '')
            })
        print(f"  📄 Loaded {len(training_examples) - start_count} examples from files")
    
    # Load memory
    if source in ["memories", "all"]:
        start_count = len(training_examples)
        memory_records = load_memory()
        for record in memory_records:
            if not record.get('content'):
                continue
            record_type = record.get('type', '')
            context = record.get('context', 'memory')
            instruction = f"Recall {record_type}: {context}" if record_type else f"Recall: {context}"
            content = record.get('content', '')
            if not isinstance(content, str):
                content = json.dumps(content)
            training_examples.append({
                'instruction': instruction,
                'response': content
            })
        print(f"  🧠 Loaded {len(training_examples) - start_count} examples from memory")
    
    # Shuffle and write
    random.shuffle(training_examples)
    
    valid_count = 0
    with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
        for example in training_examples:
            try:
                # Clean text
                cleaned = {}
                for key, value in example.items():
                    if isinstance(value, str):
                        cleaned[key] = value.encode('utf-8', errors='replace').decode('utf-8')
                        cleaned[key] = cleaned[key].replace('\x00', '').replace('\ufffd', '')
                    else:
                        cleaned[key] = value
                
                json_str = json.dumps(cleaned, ensure_ascii=False)
                json.loads(json_str)  # Validate
                f.write(json_str + '\n')
                valid_count += 1
            except Exception as e:
                continue
    
    print(f"  ✅ Wrote {valid_count} examples to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPSPEED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_deepspeed_config(
    output_dir: Path,
    num_gpus: int = 1,
    use_fp16: bool = True,
    gradient_accumulation_steps: int = 16,
) -> Path:
    """
    Create DeepSpeed ZeRO-3 configuration optimized for CPU offloading.
    
    Key settings for 20B model on limited VRAM:
    - Stage 3: Partition optimizer, gradients, AND parameters
    - CPU offload for both parameters and optimizer
    - Small bucket sizes to reduce peak GPU memory
    - Conservative live parameters limit
    """
    # Effective batch size = micro_batch * grad_accum * num_gpus
    micro_batch = 1
    train_batch_size = micro_batch * gradient_accumulation_steps * num_gpus
    
    config = {
        # Batch sizes
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        
        # ZeRO-3 with aggressive CPU offloading
        "zero_optimization": {
            "stage": 3,
            
            # Offload parameters to CPU (critical for large models)
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "buffer_size": 1e8,  # 100M elements per buffer
            },
            
            # Offload optimizer to CPU (saves massive GPU memory)
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False,  # Slower but more stable
            },
            
            # Communication optimization
            "overlap_comm": True,
            "contiguous_gradients": True,
            
            # Memory management - keep these small for large models
            "reduce_bucket_size": 5e7,           # 50M (smaller = less peak memory)
            "stage3_prefetch_bucket_size": 5e7,  # 50M
            "stage3_param_persistence_threshold": 1e5,  # 100K params stay on GPU
            
            # Critical: limit live parameters on GPU
            "stage3_max_live_parameters": 1e8,   # 100M max at once
            "stage3_max_reuse_distance": 1e8,    # 100M
            
            # For saving full weights
            "stage3_gather_16bit_weights_on_model_save": True,
            
            # Sub-group size for memory efficiency
            "sub_group_size": 1e9,
        },
        
        # FP16 configuration
        # Note: P40 supports FP16 but without tensor cores
        # Still useful for memory savings (half the size)
        "fp16": {
            "enabled": use_fp16,
            "loss_scale": 0,  # Dynamic loss scaling
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        
        # Disable BF16 (not supported on Pascal)
        "bf16": {
            "enabled": False,
        },
        
        # Optimizer - let Trainer handle this
        "zero_allow_untested_optimizer": True,
        
        # Misc
        "wall_clock_breakdown": False,
        "steps_per_print": 100,
    }
    
    config_path = output_dir / "ds_config_zero3.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


# ═══════════════════════════════════════════════════════════════════════════════
# CPU-ONLY TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cpu_only(
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    config: Config,
    resume_from_checkpoint: Optional[str] = None,
    log: Optional[logging.Logger] = None,
):
    """
    CPU-only training with manual training loop.
    No GPU memory requirements, but slower.
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    if log is None:
        log = logging.getLogger("finetune")
    
    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Disable CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    log.info("🖥️  CPU-only training mode")
    log.info(f"   This will be slow but memory-efficient")
    log.info("")
    
    # Import ML libraries
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    
    # Load tokenizer
    log.info("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to CPU
    log.info("📦 Loading model to CPU (this takes a while for large models)...")
    log.info("   Tip: Watch memory with 'htop' in another terminal")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # FP32 for CPU (more stable)
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    log.info("   ✅ Model loaded!")
    
    # Check for checkpoint
    global_step = 0
    saved_optimizer_state = None
    saved_scheduler_state = None
    
    # Find checkpoint
    checkpoint_path = None
    if resume_from_checkpoint:
        cp = Path(resume_from_checkpoint)
        if not cp.is_absolute():
            cp = output_dir / resume_from_checkpoint
        if cp.exists():
            checkpoint_path = cp
    
    if not checkpoint_path:
        checkpoints = sorted(glob.glob(str(output_dir / "checkpoint-*")))
        if checkpoints:
            checkpoint_path = Path(checkpoints[-1])
    
    # Apply LoRA or load from checkpoint
    if checkpoint_path and checkpoint_path.exists():
        log.info(f"📂 Loading checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        
        # Extract step
        if "checkpoint-" in checkpoint_path.name:
            try:
                global_step = int(checkpoint_path.name.split("checkpoint-")[1])
            except:
                pass
        
        # Load optimizer/scheduler state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            log.info("   Loading optimizer/scheduler state...")
            state = torch.load(state_path, map_location="cpu")
            saved_optimizer_state = state.get("optimizer")
            saved_scheduler_state = state.get("scheduler")
            if "global_step" in state:
                global_step = state["global_step"]
        
        log.info(f"   Resuming from step {global_step}")
    else:
        log.info("🔧 Applying LoRA adapters...")
        for param in model.parameters():
            param.requires_grad = False
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # Load dataset
    log.info(f"📊 Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    def tokenize_fn(examples):
        texts = [
            f"### Instruction:\n{inst}\n\n### Response:\n{resp}{tokenizer.eos_token}"
            for inst, resp in zip(examples["instruction"], examples["response"])
        ]
        tok = tokenizer(texts, truncation=True, max_length=config.max_length, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok
    
    log.info("   Tokenizing...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=(global_step == 0),  # Don't shuffle when resuming
        num_workers=0,
    )
    
    # Setup optimizer and scheduler
    log.info("⚙️  Setting up optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
    
    total_steps = len(dataloader) * config.epochs
    warmup_steps = min(config.warmup_steps, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Restore optimizer/scheduler state
    if saved_optimizer_state:
        log.info("   Restoring optimizer state...")
        optimizer.load_state_dict(saved_optimizer_state)
    if saved_scheduler_state:
        log.info("   Restoring scheduler state...")
        scheduler.load_state_dict(saved_scheduler_state)
    elif global_step > 0:
        log.info(f"   Advancing scheduler by {global_step} steps...")
        for _ in range(global_step):
            scheduler.step()
    
    # Calculate training info
    steps_per_epoch = len(dataloader)
    start_epoch = global_step // steps_per_epoch
    start_step_in_epoch = global_step % steps_per_epoch
    remaining_steps = total_steps - global_step
    
    log.info("")
    log.info(f"📈 Training plan:")
    log.info(f"   Dataset size: {len(tokenized)} examples")
    log.info(f"   Steps per epoch: {steps_per_epoch}")
    log.info(f"   Total epochs: {config.epochs}")
    log.info(f"   Starting from: epoch {start_epoch + 1}, step {start_step_in_epoch}")
    log.info(f"   Remaining steps: {remaining_steps}")
    log.info("")
    
    # Training loop
    log.info("🚀 Starting training...")
    model.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config.epochs):
        log.info(f"\n{'═' * 50}")
        log.info(f"  Epoch {epoch + 1}/{config.epochs}")
        log.info(f"{'═' * 50}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Recreate dataloader with shuffle after first epoch
        if epoch > start_epoch:
            dataloader = DataLoader(tokenized, batch_size=config.batch_size, shuffle=True, num_workers=0)
        
        # Progress bar
        if epoch == start_epoch and start_step_in_epoch > 0:
            # Skip to checkpoint position
            skip_bar = tqdm(total=start_step_in_epoch, desc="⏩ Skipping to checkpoint", unit="batch")
            
            for idx, batch in enumerate(dataloader):
                if idx < start_step_in_epoch:
                    skip_bar.update(1)
                    continue
                
                if idx == start_step_in_epoch:
                    skip_bar.close()
                    log.info(f"   Resuming training at batch {start_step_in_epoch}...")
                    progress = tqdm(
                        total=steps_per_epoch - start_step_in_epoch,
                        desc=f"  Training",
                        unit="batch"
                    )
                
                # Training step
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                progress.update(1)
                
                # Checkpoint
                if global_step % config.checkpoint_steps == 0:
                    _save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, epoch_loss, output_dir, log)
            
            progress.close()
        else:
            # Normal epoch
            progress = tqdm(dataloader, total=steps_per_epoch, desc=f"  Training", unit="batch")
            
            for batch in progress:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                
                # Checkpoint
                if global_step % config.checkpoint_steps == 0:
                    _save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, epoch_loss, output_dir, log)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        log.info(f"\n   Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
    
    # Save final model
    log.info(f"\n💾 Saving final model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("✅ Training complete!")


def _save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, epoch_loss, output_dir, log):
    """Save a checkpoint with full state."""
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    
    import torch
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'epoch_loss': epoch_loss,
    }, ckpt_dir / 'training_state.pt')
    
    log.info(f"\n   💾 Checkpoint saved: {ckpt_dir.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU + CPU OFFLOAD TRAINING (DeepSpeed ZeRO-3)
# ═══════════════════════════════════════════════════════════════════════════════

def train_gpu_with_offload(
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    config: Config,
    resume_from_checkpoint: Optional[str] = None,
    log: Optional[logging.Logger] = None,
):
    """
    GPU training with CPU offload using Accelerate + DeepSpeed ZeRO-3.
    
    Uses Accelerate's DeepSpeed integration which properly handles:
    - Model partitioning during loading
    - CPU offloading of parameters and optimizer
    - FP16 training
    """
    import torch
    
    if log is None:
        log = logging.getLogger("finetune")
    
    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Get GPU info
    gpu_info = get_gpu_info()
    num_gpus = gpu_info["count"]
    
    if num_gpus == 0:
        log.warning("No GPUs detected! Falling back to CPU-only mode.")
        return train_cpu_only(model_id, dataset_path, output_dir, config, resume_from_checkpoint, log)
    
    log.info(f"🎮 GPU + CPU Offload mode (Accelerate + DeepSpeed ZeRO-3)")
    log.info(f"   {num_gpus} GPU(s) detected:")
    for i, dev in enumerate(gpu_info["devices"]):
        tensor_cores = "✅ Tensor Cores" if dev["has_tensor_cores"] else "❌ No Tensor Cores"
        log.info(f"   [{i}] {dev['name']} - {dev['memory_total_gb']}GB - CC {dev['compute_capability']} - {tensor_cores}")
    log.info("")
    
    # Check for Pascal GPUs (no tensor cores)
    has_tensor_cores = any(dev["has_tensor_cores"] for dev in gpu_info["devices"])
    use_fp16 = True  # We'll use FP16 for memory savings
    
    if not has_tensor_cores:
        log.warning("⚠️  Pascal GPU detected (no Tensor Cores)")
        log.warning("   Using FP16 for memory efficiency")
        log.warning("   Will patch DeepSpeed to allow FP16 on Pascal")
        log.info("")
        
        # Monkey-patch DeepSpeed's sanity check
        try:
            import deepspeed.runtime.engine as ds_engine
            original_sanity_check = ds_engine.DeepSpeedEngine._do_sanity_check
            
            def patched_sanity_check(self):
                try:
                    original_sanity_check(self)
                except ValueError as e:
                    if "fp16 is not supported" in str(e).lower():
                        pass  # Ignore FP16 check on Pascal
                    else:
                        raise
            
            ds_engine.DeepSpeedEngine._do_sanity_check = patched_sanity_check
            log.info("   ✅ Patched DeepSpeed to allow FP16 on Pascal")
        except Exception as e:
            log.warning(f"   Could not patch DeepSpeed: {e}")
    
    # Memory warning for long sequences
    if config.max_length >= 2048:
        total_vram = sum(dev["memory_total_gb"] for dev in gpu_info["devices"])
        log.warning(f"⚠️  Long sequences ({config.max_length} tokens) with {total_vram}GB total VRAM")
        log.warning("   If OOM occurs, try: --cpu_only or --max_length 1024")
        log.info("")
    
    # STEP 1: Initialize distributed if needed
    import torch.distributed as dist
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if not dist.is_initialized() and world_size > 1:
        import deepspeed
        log.info("   Initializing DeepSpeed distributed...")
        deepspeed.init_distributed()
    
    log.info(f"   Rank {local_rank + 1}/{world_size}")
    
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    
    # Create DeepSpeed config
    ds_config_path = create_deepspeed_config(
        output_dir=output_dir,
        num_gpus=num_gpus,
        use_fp16=use_fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    log.info(f"📋 Created DeepSpeed config: {ds_config_path}")
    
    # Import after distributed init
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    
    # Create Accelerator with DeepSpeed
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=str(ds_config_path),
        zero3_init_flag=True,  # Enable ZeRO-3 init
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        zero_stage=3,
    )
    
    accelerator = Accelerator(
        mixed_precision="fp16" if use_fp16 else "no",
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    log.info("   ✅ Accelerator initialized with DeepSpeed ZeRO-3")
    
    # Load tokenizer
    log.info("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # STEP 2: Load model - Accelerate handles ZeRO-3 partitioning
    log.info("📦 Loading model with ZeRO-3 partitioning...")
    log.info("   Accelerate will partition during loading")
    log.info("   This may take 5-10 minutes for 20B+ models...")
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Let Accelerate handle the model loading with proper ZeRO-3 init
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    
    log.info("   ✅ Model loaded!")
    
    # Check for checkpoint
    checkpoint_path = None
    if resume_from_checkpoint:
        cp = Path(resume_from_checkpoint)
        if not cp.is_absolute():
            cp = output_dir / resume_from_checkpoint
        if cp.exists():
            checkpoint_path = cp
    
    if not checkpoint_path:
        checkpoints = sorted(glob.glob(str(output_dir / "checkpoint-*")))
        if checkpoints:
            checkpoint_path = Path(checkpoints[-1])
    
    # Apply LoRA
    if checkpoint_path and checkpoint_path.exists():
        log.info(f"📂 Loading LoRA checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, str(checkpoint_path), is_trainable=True)
    else:
        log.info("🔧 Applying LoRA adapters...")
        for param in model.parameters():
            param.requires_grad = False
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        log.info("   ✅ Gradient checkpointing enabled")
    
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # Load dataset
    log.info(f"📊 Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    def tokenize_fn(examples):
        texts = []
        for inst, resp in zip(examples["instruction"], examples["response"]):
            inst_str = " ".join(str(x) for x in inst) if isinstance(inst, list) else str(inst or "")
            resp_str = " ".join(str(x) for x in resp) if isinstance(resp, list) else str(resp or "")
            text = f"### Instruction:\n{inst_str}\n\n### Response:\n{resp_str}{tokenizer.eos_token}"
            texts.append(text)
        tok = tokenizer(texts, truncation=True, max_length=config.max_length, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    log.info(f"   Dataset size: {len(tokenized)} examples")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create optimizer (only for trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)
    
    # Calculate total steps
    num_update_steps = len(dataloader) // config.gradient_accumulation_steps * config.epochs
    
    # Create scheduler
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_update_steps,
    )
    
    # Prepare with Accelerate - this handles DeepSpeed partitioning
    log.info("🚀 Preparing model with Accelerate...")
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    log.info("   ✅ Model prepared and partitioned!")
    
    # Training loop
    log.info("")
    log.info("🚀 Starting training...")
    
    global_step = 0
    start_epoch = 0
    
    # Handle checkpoint resumption
    if checkpoint_path:
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            global_step = state.get("global_step", 0)
            start_epoch = state.get("epoch", 0)
            log.info(f"   Resuming from step {global_step}, epoch {start_epoch}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, config.epochs):
        log.info(f"\n{'═' * 50}")
        log.info(f"  Epoch {epoch + 1}/{config.epochs}")
        log.info(f"{'═' * 50}")
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"  Training", unit="batch", disable=(not accelerator.is_main_process))
        
        for batch in progress:
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if accelerator.sync_gradients:
                global_step += 1
            
            if accelerator.is_main_process:
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
            
            # Checkpoint
            if global_step > 0 and global_step % config.checkpoint_steps == 0 and accelerator.is_main_process:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                
                # Save LoRA weights (unwrap model first)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                
                # Save training state
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                }, ckpt_dir / 'training_state.pt')
                
                log.info(f"\n   💾 Checkpoint saved: {ckpt_dir.name}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if accelerator.is_main_process:
            log.info(f"\n   Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
    
    # Save final model
    if accelerator.is_main_process:
        log.info(f"\n💾 Saving final model to {output_dir}...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log.info("✅ Training complete!")
    
    accelerator.wait_for_everyone()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model
    parser.add_argument('--model_name', type=str, default='gpt-oss:20b',
                        help='Model name (HuggingFace ID or Ollama-style)')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace token (or set HF_TOKEN env)')
    
    # Data
    parser.add_argument('--dataset_source', type=str, 
                        choices=['conversations', 'files', 'memories', 'all'],
                        default='all', help='Data source')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Use existing dataset file')
    parser.add_argument('--export_only', action='store_true',
                        help='Only export dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=1, help='Micro batch size')
    parser.add_argument('--gradient_accumulation', type=int, default=16, help='Gradient accumulation steps')
    
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    
    # Mode
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU-only training')
    
    # Checkpointing
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output adapter name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from checkpoint (e.g., checkpoint-1500)')
    
    # DeepSpeed (auto-added by launcher)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank (set by DeepSpeed)')
    
    args = parser.parse_args()
    
    # Handle DeepSpeed local_rank
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Setup HF token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Generate output name
    if args.output_name is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        model_safe = args.model_name.replace(':', '-').replace('/', '-')
        args.output_name = f"lora-{model_safe}-{timestamp}"
    
    output_dir = ADAPTERS_DIR / args.output_name
    
    # Setup logging
    log = setup_logging(output_dir, rank=local_rank)
    
    # Only rank 0 prints banner
    if local_rank == 0:
        print_banner("LoRA Fine-Tuning")
        
        # Config
        print("\033[1m📋 Configuration:\033[0m")
        print_config_table({
            "Model": args.model_name,
            "Dataset": args.dataset_source,
            "Epochs": args.epochs,
            "Learning Rate": args.learning_rate,
            "Max Length": args.max_length,
            "Mode": "CPU-only" if args.cpu_only else "GPU + CPU Offload",
            "Output": args.output_name,
        })
        
        # GPU info
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            print(f"\033[1m🎮 GPUs:\033[0m")
            for i, dev in enumerate(gpu_info["devices"]):
                print(f"  [{i}] {dev['name']} ({dev['memory_total_gb']}GB)")
            print()
        else:
            print("\033[1m🖥️  No GPUs available - will use CPU\033[0m\n")
    
    # Create config object
    config = Config(
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Export dataset (only rank 0)
    if local_rank == 0:
        if args.dataset_path:
            dataset_path = Path(args.dataset_path)
            if not dataset_path.exists():
                log.error(f"Dataset not found: {dataset_path}")
                return 1
            log.info(f"📂 Using existing dataset: {dataset_path}")
        else:
            log.info("📦 Exporting training data...")
            dataset_path = export_training_data(source=args.dataset_source)
        
        if args.export_only:
            log.info("\n✅ Dataset exported. Use --dataset_path to train later.")
            return 0
    else:
        # Non-rank-0 processes wait for dataset
        import time
        while args.dataset_path is None:
            # Find latest dataset
            datasets = sorted(TRAINING_DATA_DIR.glob("dataset-*.jsonl"))
            if datasets:
                args.dataset_path = str(datasets[-1])
                break
            time.sleep(1)
        dataset_path = Path(args.dataset_path)
    
    # Get model ID
    model_id = get_model_id(args.model_name)
    
    # Train
    if args.cpu_only:
        train_cpu_only(
            model_id=model_id,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
            resume_from_checkpoint=args.resume_from_checkpoint,
            log=log,
        )
    else:
        train_gpu_with_offload(
            model_id=model_id,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config,
            resume_from_checkpoint=args.resume_from_checkpoint,
            log=log,
        )
    
    if local_rank == 0:
        print_banner("Training Complete!", char="═")
        log.info(f"🎉 Adapter saved to: {output_dir}")
        log.info("")
        log.info("Next steps:")
        log.info(f"  • Test: ollama run {args.model_name} --lora {output_dir}")
        log.info(f"  • Merge: python merge_lora.py --adapter {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())