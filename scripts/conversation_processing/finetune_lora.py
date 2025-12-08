#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Large Language Models

Supports:
- Models up to 20B+ parameters with CPU offloading
- DeepSpeed ZeRO-3 for multi-GPU training
- Single GPU, multi-GPU, and CPU-only modes
- Checkpoint resumption with optimizer/scheduler state
- Automatic dataset preparation from conversations/files/memory
- File logging (survives SSH disconnect)

============================================================================
Dev Examples (NICK WILL DELETE)
============================================================================
# Copy script to server
# scp scripts/finetune_lora.py nick@192.168.1.77:~/ai/scripts/finetune_lora.py

# SINGLE P40 (24GB) - Use CPU-only mode for 20B model (model is 40GB, doesn't fit in 24GB):
ssh nick@192.168.1.77 "cd ~/ai && source venv/bin/activate && python scripts/finetune_lora.py --model_name gpt-oss:20b --output_name alden-gpt-oss-20b-1764689640337 --resume_from_checkpoint checkpoint-1500 --dataset_path training_data/alden-dataset-1764565413005.jsonl --max_length 2048 --cpu_only"

# TWO P40s (48GB total) - DeepSpeed ZeRO-3 shards model across GPUs:
ssh nick@192.168.1.77 "cd ~/ai && source venv/bin/activate && accelerate launch --num_processes 2 scripts/finetune_lora.py --model_name gpt-oss:20b --output_name alden-gpt-oss-20b-1764689640337 --resume_from_checkpoint checkpoint-1500 --dataset_path training_data/alden-dataset-1764565413005.jsonl --max_length 2048"

# Alternative for 2 GPUs (without accelerate):
ssh nick@192.168.1.77 "cd ~/ai && source venv/bin/activate && python scripts/finetune_lora.py --model_name gpt-oss:20b --output_name alden-gpt-oss-20b-1764689640337 --resume_from_checkpoint checkpoint-1500 --dataset_path training_data/alden-dataset-1764565413005.jsonl --max_length 2048"

# Server management commands:
# Check if training is running:
ssh nick@192.168.1.77 "ps aux | grep -E 'finetune_lora|python.*scripts' | grep -v grep"

# Kill all training processes:
ssh nick@192.168.1.77 "pkill -9 -f finetune; sleep 2; ps aux | grep python | grep -v grep | grep -v unattended"

# Check checkpoints:
ssh nick@192.168.1.77 "ls -la ~/ai/adapters/alden-gpt-oss-20b-1764689640337/"

# Check logs:
# ssh nick@192.168.1.77 "tail -f ~/ai/adapters/alden-gpt-oss-20b-1764689640337/training_*.log"
# ssh nick@192.168.1.77 "tail -50 ~/ai/adapters/alden-gpt-oss-20b-1764689640337/training_*.log"

============================================================================
USAGE EXAMPLES
============================================================================

Single GPU (24GB) - CPU-only mode for 20B+ models:
  python finetune_lora.py --model_name <model> --cpu_only --max_length 2048

Multi-GPU (2x 24GB) - DeepSpeed ZeRO-3:
  accelerate launch --num_processes 2 finetune_lora.py --model_name <model>

Resume from checkpoint:
  python finetune_lora.py --model_name <model> --resume_from_checkpoint checkpoint-1500

============================================================================
"""

import json
import os
import sys
import argparse
import random
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup module-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
FILES_DIR = PROJECT_ROOT / "files"
MEMORY_DIR = PROJECT_ROOT / "memory"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
ADAPTERS_DIR = PROJECT_ROOT / "adapters"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Ensure directories exist
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_conversations() -> List[Dict[str, Any]]:
    """Load conversations from JSON or JSONL files."""
    conversations = []
    
    # Try conversations.json first
    conversations_json = CONVERSATIONS_DIR / "conversations.json"
    if conversations_json.exists():
        log.info(f"Loading conversations from {conversations_json}...")
        with open(conversations_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            conversations = data if isinstance(data, list) else [data]
        log.info(f"Loaded {len(conversations)} conversations")
        return conversations
    
    # Load from JSONL files
    jsonl_files = sorted(CONVERSATIONS_DIR.glob('conversation_*.jsonl'))
    if jsonl_files:
        log.info(f"Loading from {len(jsonl_files)} JSONL files...")
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
                                    'timestamp': msg.get('timestamp', '')
                                })
                        except json.JSONDecodeError:
                            continue
                    
                    if messages:
                        conv_id = jsonl_file.stem.replace('conversation_', '')
                        conversations.append({'id': conv_id, 'messages': messages})
            except Exception as e:
                log.warning(f"Error loading {jsonl_file}: {e}")
        
        log.info(f"Loaded {len(conversations)} conversations")
        return conversations
    
    log.warning("No conversations found")
    return conversations


def load_files() -> List[Dict[str, Any]]:
    """Load text files from the files directory."""
    files = []
    
    if not FILES_DIR.exists():
        return files
    
    for ext in ['*.txt', '*.md']:
        for file_path in FILES_DIR.rglob(ext):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                rel_path = file_path.relative_to(FILES_DIR)
                title = file_path.stem
                
                # Extract title from markdown heading if available
                if content.startswith('# '):
                    first_line = content.split('\n')[0]
                    if first_line.startswith('# '):
                        title = first_line[2:].strip()
                
                files.append({
                    'filepath': str(rel_path),
                    'filename': file_path.name,
                    'title': title,
                    'content': content
                })
            except Exception as e:
                log.warning(f"Error loading {file_path}: {e}")
    
    log.info(f"Loaded {len(files)} files")
    return files


def load_memory() -> List[Dict[str, Any]]:
    """Load memory records from JSONL files."""
    memory_records = []
    
    if not MEMORY_DIR.exists():
        return memory_records
    
    jsonl_files = list(MEMORY_DIR.glob('*.jsonl'))
    for jsonl_file in jsonl_files:
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
            log.warning(f"Error reading {jsonl_file}: {e}")
    
    log.info(f"Loaded {len(memory_records)} memory records from {len(jsonl_files)} files")
    return memory_records


def export_training_data(
    source: str = "all",
    output_path: Optional[Path] = None
) -> Path:
    """Export training data to JSONL format."""
    if output_path is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        output_path = TRAINING_DATA_DIR / f"training-dataset-{timestamp}.jsonl"
    
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_examples = []
    
    # Load conversations
    if source in ["conversations", "all"]:
        conversations = load_conversations()
        for conv in conversations:
            messages = conv.get('messages', [])
            for i in range(len(messages) - 1):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if (user_msg.get('role') == 'user' and 
                    assistant_msg.get('role') == 'assistant'):
                    training_examples.append({
                        'instruction': user_msg.get('content', ''),
                        'response': assistant_msg.get('content', '')
                    })
        log.info(f"Added {len(training_examples)} conversation examples")
    
    # Load files
    if source in ["files", "all"]:
        start_count = len(training_examples)
        files = load_files()
        for file_data in files:
            title = file_data.get('title', file_data.get('filename', '').replace('.txt', '').replace('.md', ''))
            training_examples.append({
                'instruction': f"Read this transmission: {title}",
                'response': file_data.get('content', '')
            })
        log.info(f"Added {len(training_examples) - start_count} file examples")
    
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
        log.info(f"Added {len(training_examples) - start_count} memory examples")
    
    # Shuffle and write
    random.shuffle(training_examples)
    log.info(f"Writing {len(training_examples)} training examples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    log.info(f"[OK] Dataset exported: {output_path}")
    return output_path


# ============================================================================
# Model Utilities
# ============================================================================

def get_huggingface_model_id(model_name: str) -> str:
    """Convert model name to HuggingFace model ID."""
    model_mapping = {
        'gpt-oss:20b': 'openai/gpt-oss-20b',
        'gpt-oss/20b': 'openai/gpt-oss-20b',
    }
    
    # Check mapping first
    if model_name in model_mapping:
        return model_mapping[model_name]
    
    # If it contains ':', assume Ollama format and convert
    if ':' in model_name:
        parts = model_name.split(':')
        if len(parts) == 2:
            model_base = parts[0].replace('-', '/')
            return f"{model_base}-{parts[1]}"
    
    # Otherwise assume it's already a HuggingFace ID or local path
    return model_name


# ============================================================================
# Training Script Generation
# ============================================================================

def create_deepspeed_config(output_dir: Path) -> Path:
    """Create DeepSpeed ZeRO-3 configuration with CPU offloading."""
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        # Concrete values required for deepspeed.zero.Init()
        # batch_size = micro_batch * grad_accum * num_gpus = 1 * 16 * 1 = 16
        "gradient_accumulation_steps": 16,
        "gradient_clipping": 1.0,
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }
    
    config_path = output_dir / "deepspeed_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def _create_cpu_training_script(
    model_id: str,
    dataset_path: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    hf_token: Optional[str],
    max_length: int,
    resume_from_checkpoint: Optional[str]
) -> str:
    """Create a CPU-only training script with manual training loop (no Trainer overhead)."""
    return f'''#!/usr/bin/env python3
"""
LoRA Fine-tuning Script (CPU-Only Mode)
Uses manual training loop for best CPU performance - no Trainer/Accelerator overhead.
"""
import os
import sys
import glob
import subprocess
import random
import logging
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Setup file logging (survives SSH disconnect)
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.log")
    
    # Create logger
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger, log_file

# Configuration
MODEL_NAME = {repr(model_id)}
DATASET_PATH = {repr(dataset_path)}
OUTPUT_DIR = {repr(output_dir)}
EPOCHS = {epochs}
LEARNING_RATE = {learning_rate}
MAX_LENGTH = {max_length}
HF_TOKEN = {repr(hf_token) if hf_token else "None"}
RESUME_CHECKPOINT = {repr(resume_from_checkpoint) if resume_from_checkpoint else "None"}

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Initialize logging first
log, log_file = setup_logging(OUTPUT_DIR)

# Kill any stale training processes before starting
def cleanup_stale_processes():
    try:
        my_pid = os.getpid()
        result = subprocess.run(
            ["pgrep", "-f", "finetune.*\\\\.py"],
            capture_output=True, text=True
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split("\\n"):
                pid = int(pid.strip())
                if pid != my_pid:
                    log.info(f"Killing stale training process: {{pid}}")
                    try:
                        os.kill(pid, 9)
                    except:
                        pass
    except Exception as e:
        pass  # pgrep might not exist on all systems

log.info("Cleaning up stale processes...")
cleanup_stale_processes()

# Disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ACCELERATE_USE_CPU"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

log.info("="*70)
log.info("CPU-Only LoRA Fine-Tuning (Manual Loop)")
log.info("="*70)
log.info(f"Model: {{MODEL_NAME}}")
log.info(f"Dataset: {{DATASET_PATH}}")
log.info(f"Output: {{OUTPUT_DIR}}")
log.info(f"Epochs: {{EPOCHS}} | LR: {{LEARNING_RATE}} | Max Length: {{MAX_LENGTH}}")
log.info(f"Log file: {{log_file}}")
log.info(f"Random seed: {{SEED}}")
log.info("="*70)

# Load tokenizer
log.info("\\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Patch CUDA functions for MXFP4 quantizer (required for CPU-only)
log.info("\\nLoading model (this takes a while for large models)...")
original_get_device_capability = torch.cuda.get_device_capability
original_get_device_properties = torch.cuda.get_device_properties

def patched_get_device_capability(device=None):
    return (7, 0) if torch.cuda.device_count() == 0 else original_get_device_capability(device)

def patched_get_device_properties(device):
    if torch.cuda.device_count() == 0:
        class MockProps:
            name = "CPU"; major = 7; minor = 0; total_memory = 0
        return MockProps()
    return original_get_device_properties(device)

torch.cuda.get_device_capability = patched_get_device_capability
torch.cuda.get_device_properties = patched_get_device_properties

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
finally:
    torch.cuda.get_device_capability = original_get_device_capability
    torch.cuda.get_device_properties = original_get_device_properties

log.info("Model loaded in bfloat16 (~40GB)")

# Check for checkpoint
resume_path = None
global_step = 0
saved_optimizer_state = None
saved_scheduler_state = None

if RESUME_CHECKPOINT:
    check_path = RESUME_CHECKPOINT if os.path.isabs(RESUME_CHECKPOINT) else os.path.join(OUTPUT_DIR, RESUME_CHECKPOINT)
    if os.path.exists(check_path):
        resume_path = check_path
if not resume_path:
    checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if checkpoints:
        resume_path = checkpoints[-1]

# Apply LoRA or load checkpoint
if resume_path:
    log.info(f"\\nLoading checkpoint: {{resume_path}}")
    model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
    # Extract step from checkpoint name
    ckpt_name = os.path.basename(resume_path)
    if "checkpoint-" in ckpt_name:
        try:
            global_step = int(ckpt_name.split("checkpoint-")[1])
            log.info(f"Resuming from step {{global_step}}")
        except:
            pass
    
    # Load optimizer/scheduler state if available
    training_state_path = os.path.join(resume_path, "training_state.pt")
    if os.path.exists(training_state_path):
        log.info(f"Loading optimizer/scheduler state from checkpoint...")
        training_state = torch.load(training_state_path, map_location="cpu")
        saved_optimizer_state = training_state.get("optimizer")
        saved_scheduler_state = training_state.get("scheduler")
        # Prefer step from saved state if available (more accurate)
        if "global_step" in training_state:
            global_step = training_state["global_step"]
            log.info(f"Restored global_step={{global_step}} from training state")
else:
    log.info("\\nApplying LoRA adapters...")
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# Load and tokenize dataset
log.info(f"\\nLoading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH)

def tokenize_fn(examples):
    texts = [f"### Instruction:\\n{{i}}\\n\\n### Response:\\n{{r}}<|endoftext|>" 
             for i, r in zip(examples["instruction"], examples["response"])]
    tok = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok

log.info("Tokenizing...")
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

dataloader = DataLoader(tokenized["train"], batch_size=1, shuffle=True, num_workers=0)

# Optimizer and scheduler
log.info("\\nSetting up optimizer...")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(dataloader) * EPOCHS
warmup_steps = min(100, total_steps // 10)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Restore optimizer/scheduler state if resuming (preserves momentum and LR schedule)
if saved_optimizer_state is not None:
    log.info("Restoring optimizer state (preserves AdamW momentum)...")
    optimizer.load_state_dict(saved_optimizer_state)
if saved_scheduler_state is not None:
    log.info("Restoring scheduler state (skips warmup correctly)...")
    scheduler.load_state_dict(saved_scheduler_state)
elif global_step > 0:
    # If no saved scheduler state but resuming, manually advance scheduler
    log.info(f"No saved scheduler state - advancing scheduler by {{global_step}} steps...")
    for _ in range(global_step):
        scheduler.step()

# Calculate remaining training
steps_per_epoch = len(dataloader)
start_epoch = global_step // steps_per_epoch
start_step_in_epoch = global_step % steps_per_epoch
remaining_steps = total_steps - global_step
log.info(f"\\nTotal steps: {{total_steps}}, Starting from step {{global_step}}")
log.info(f"Steps per epoch: {{steps_per_epoch}}, Remaining: {{remaining_steps}}")
log.info(f"Current LR: {{scheduler.get_last_lr()[0]:.2e}}")

# Training loop
log.info("\\nStarting training...")
model.train()
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(start_epoch, EPOCHS):
    log.info(f"\\nEpoch {{epoch + 1}}/{{EPOCHS}}")
    epoch_loss = 0.0
    num_batches = 0
    
    # Calculate correct progress bar range for this epoch
    if epoch == start_epoch and start_step_in_epoch > 0:
        # Resuming mid-epoch: skip batches with progress bar, then train remaining
        remaining_in_epoch = steps_per_epoch - start_step_in_epoch
        log.info(f"Skipping first {{start_step_in_epoch}} batches, training on remaining {{remaining_in_epoch}}")
        
        # Skip batches with visible progress
        dataloader_iter = iter(dataloader)
        for _ in tqdm(range(start_step_in_epoch), desc="Skipping", leave=False):
            next(dataloader_iter)
        
        # Now train on remaining batches - show absolute position (1500/9701)
        progress = tqdm(dataloader_iter, total=steps_per_epoch, initial=start_step_in_epoch, desc=f"Epoch {{epoch+1}}")
    else:
        progress = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {{epoch+1}}")
    
    for batch in progress:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        progress.set_postfix({{"loss": f"{{loss.item():.4f}}", "avg": f"{{epoch_loss/num_batches:.4f}}", "lr": f"{{scheduler.get_last_lr()[0]:.2e}}"}})
        
        # Checkpoint every 500 steps
        if global_step % 500 == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{{global_step}}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # Save optimizer and scheduler state for proper resumption
            torch.save({{
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'epoch_loss': epoch_loss,
            }}, os.path.join(ckpt_dir, 'training_state.pt'))
            log.info(f"\\nSaved checkpoint: {{ckpt_dir}} (with optimizer/scheduler state)")
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    log.info(f"Epoch {{epoch+1}} complete. Avg loss: {{avg_loss:.4f}}")

# Save final model
log.info(f"\\nSaving final model to {{OUTPUT_DIR}}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
log.info("\\nTraining complete!")
log.info(f"Log saved to: {{log_file}}")
'''


def create_training_script(
    model_name: str,
    dataset_path: Path,
    output_name: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    hf_token: Optional[str] = None,
    max_length: int = 256,
    cpu_only: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    use_deepspeed: bool = True
) -> Path:
    """Create the training script."""
    model_id = get_huggingface_model_id(model_name)
    output_dir = ADAPTERS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = SCRIPTS_DIR / f"finetune_{output_name}_{int(datetime.now().timestamp() * 1000)}.py"
    
    # For CPU-only mode, use manual training loop (no Trainer overhead)
    if cpu_only:
        script_content = _create_cpu_training_script(
            model_id, str(dataset_path), str(output_dir), epochs, learning_rate, 
            hf_token, max_length, resume_from_checkpoint
        )
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        log.info(f"[OK] Created CPU training script: {script_path}")
        return script_path
    
    # Determine if we need DeepSpeed (GPU mode)
    needs_deepspeed = use_deepspeed
    
    # Create DeepSpeed config if needed
    deepspeed_config_path = None
    if needs_deepspeed:
        try:
            deepspeed_config_path = create_deepspeed_config(output_dir)
            log.info(f"[INFO] Created DeepSpeed ZeRO-3 config: {deepspeed_config_path}")
        except Exception as e:
            log.warning(f"Could not create DeepSpeed config: {e}")
            needs_deepspeed = False
    
    # Generate GPU script content
    script_content = f'''#!/usr/bin/env python3
"""
LoRA Fine-tuning Script (Auto-generated)
Supports CPU-only, single GPU, and multi-GPU training with DeepSpeed ZeRO-3.
"""

import os
import sys
import json
import gc
import glob
import random
import logging
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Setup file logging (survives SSH disconnect)
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.log")
    
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger, log_file

# Configuration
MODEL_NAME = {repr(model_id)}
DATASET_PATH = {repr(str(dataset_path))}
OUTPUT_DIR = {repr(str(output_dir))}
EPOCHS = {epochs}
LEARNING_RATE = {learning_rate}
MAX_LENGTH = {max_length}
HF_TOKEN = {repr(hf_token) if hf_token else "None"}
RESUME_CHECKPOINT = {repr(resume_from_checkpoint) if resume_from_checkpoint else "None"}
USE_DEEPSPEED = {needs_deepspeed}
DEEPSPEED_CONFIG = {repr(str(deepspeed_config_path)) if deepspeed_config_path else "None"}

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize logging
log, log_file = setup_logging(OUTPUT_DIR)

# Check DeepSpeed availability
ds_config = None
if USE_DEEPSPEED and DEEPSPEED_CONFIG and os.path.exists(DEEPSPEED_CONFIG):
    try:
        import deepspeed
        with open(DEEPSPEED_CONFIG, 'r') as f:
            ds_config = json.load(f)
        num_gpus = torch.cuda.device_count()
        log.info(f"DeepSpeed ZeRO-3 enabled ({{num_gpus}} GPU(s))")
    except Exception as e:
        log.info(f"DeepSpeed not available: {{e}}")
        USE_DEEPSPEED = False
        DEEPSPEED_CONFIG = None
        ds_config = None

log.info("="*70)
log.info("LoRA Fine-Tuning (GPU Mode)")
log.info("="*70)
log.info(f"Model: {{MODEL_NAME}}")
log.info(f"Dataset: {{DATASET_PATH}}")
log.info(f"Output: {{OUTPUT_DIR}}")
log.info(f"Epochs: {{EPOCHS}} | LR: {{LEARNING_RATE}} | Max Length: {{MAX_LENGTH}}")
log.info(f"Mode: GPU" + (" + DeepSpeed ZeRO-3" if USE_DEEPSPEED else ""))
log.info(f"Log file: {{log_file}}")
log.info(f"Random seed: {{SEED}}")
log.info("="*70)

# Patch torch.xpu to avoid AttributeError in MXFP4 quantizer validation
original_xpu = getattr(torch, 'xpu', None)
if not hasattr(torch, 'xpu'):
    class MockXPU:
        @staticmethod
        def is_available():
            return False
    torch.xpu = MockXPU()

# Load tokenizer
log.info("\\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Check for checkpoints
resume_from_checkpoint = None
if RESUME_CHECKPOINT:
    checkpoint_path = Path(RESUME_CHECKPOINT)
    resume_from_checkpoint = str(checkpoint_path) if checkpoint_path.is_absolute() else str(Path(OUTPUT_DIR) / RESUME_CHECKPOINT)
    if not os.path.exists(resume_from_checkpoint):
        resume_from_checkpoint = None

if not resume_from_checkpoint:
    checkpoint_dirs = sorted(glob.glob(str(Path(OUTPUT_DIR) / "checkpoint-*")))
    if checkpoint_dirs:
        resume_from_checkpoint = checkpoint_dirs[-1]

# Model loading
log.info("\\nLoading model...")

model_kwargs = {{
    "trust_remote_code": True,
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
}}

if USE_DEEPSPEED and ds_config is not None:
    # DeepSpeed mode - load to CPU, DeepSpeed handles GPU distribution
    log.info("  Mode: DeepSpeed ZeRO-3")
    model_kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    log.info("  Model loaded to CPU (DeepSpeed will handle GPU sharding)")
else:
    # Single GPU - use device_map with memory limits
    log.info("  Mode: Single GPU with CPU offloading")
    max_memory = {{0: "14GiB", "cpu": "100GiB"}}
    if {max_length} >= 2048:
        max_memory[0] = "10GiB"
    elif {max_length} >= 1024:
        max_memory[0] = "12GiB"
    
    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = max_memory
    model_kwargs["offload_buffers"] = True
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    log.info("  Model loaded successfully")

# Restore torch.xpu
if original_xpu is None:
    if hasattr(torch, 'xpu'):
        delattr(torch, 'xpu')
else:
    torch.xpu = original_xpu

# Apply LoRA or load from checkpoint
if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
    log.info(f"\\nLoading checkpoint: {{resume_from_checkpoint}}")
    model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
else:
    log.info("\\nApplying LoRA adapters...")
    for param in model.parameters():
        param.requires_grad = False
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

# CRITICAL: Ensure model is in training mode
model.train()

# Ensure base model is frozen and LoRA parameters are trainable
log.info("\\nConfiguring gradients...")
lora_enabled = 0
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
        lora_enabled += 1
    else:
        param.requires_grad = False

trainable_params = [p for p in model.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters found!")

model.print_trainable_parameters()

# Configure for gradient checkpointing
if hasattr(model.config, 'use_cache'):
    model.config.use_cache = False

if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    if hasattr(model, 'get_input_embeddings'):
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# Load and tokenize dataset
log.info("\\nLoading and tokenizing dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def tokenize_function(examples):
    texts = []
    for inst, resp in zip(examples["instruction"], examples["response"]):
        inst_str = " ".join(str(x) for x in inst) if isinstance(inst, list) else str(inst or "")
        resp_str = " ".join(str(x) for x in resp) if isinstance(resp, list) else str(resp or "")
        texts.append(f"### Instruction:\\n{{inst_str}}\\n\\n### Response:\\n{{resp_str}}")
    
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors=None)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")

# Training configuration (GPU mode)
log.info("\\nConfiguring training...")
log.info("  Gradient accumulation: 16, Gradient checkpointing: enabled")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=LEARNING_RATE,
    bf16=False,  # P40 doesn't support bf16
    fp16=torch.cuda.is_available(),  # Use fp16 for P40 (Pascal architecture)
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    eval_strategy="no",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    remove_unused_columns=False,
    deepspeed=DEEPSPEED_CONFIG if USE_DEEPSPEED and DEEPSPEED_CONFIG else None,
)

# Create trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset,
    data_collator=data_collator,
)

# Handle checkpoint resumption state
trainer_resume_checkpoint = None
if resume_from_checkpoint:
    trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                existing_state = json.load(f)
            
            is_valid_state = (
                len(existing_state.get('log_history', [])) > 0 or
                existing_state.get('best_metric') is not None
            )
            
            if is_valid_state:
                log.info(f"Found valid trainer_state.json - will resume from checkpoint")
                trainer_resume_checkpoint = resume_from_checkpoint
            else:
                log.info(f"Found placeholder trainer_state.json - removing it")
                os.remove(trainer_state_path)
        except Exception as e:
            log.info(f"Warning: Could not read trainer_state.json: {{e}}")
            try:
                os.remove(trainer_state_path)
            except:
                pass

# If no valid trainer state, calculate remaining training from checkpoint
if resume_from_checkpoint and trainer_resume_checkpoint is None:
    checkpoint_name = os.path.basename(resume_from_checkpoint)
    if "checkpoint-" in checkpoint_name:
        try:
            # The checkpoint number represents EXAMPLES processed (old script had no grad accum)
            # NOT steps with gradient accumulation
            checkpoint_examples = int(checkpoint_name.split("checkpoint-")[1])
            log.info(f"\\nResuming from checkpoint: {{checkpoint_examples}} examples processed")
            
            # Calculate training metrics
            dataset_len = len(tokenized_dataset["train"] if "train" in tokenized_dataset else tokenized_dataset)
            grad_accum = training_args.gradient_accumulation_steps
            steps_per_epoch = max(1, dataset_len // grad_accum)  # steps with grad accum
            
            # Convert checkpoint examples to epochs completed
            completed_epochs = float(checkpoint_examples) / dataset_len
            target_epochs = training_args.num_train_epochs
            remaining_epochs = max(0.1, target_epochs - completed_epochs)
            
            # Convert to equivalent step number for this script
            equivalent_step = int(checkpoint_examples / grad_accum)
            
            log.info(f"Dataset: {{dataset_len}} examples")
            log.info(f"Gradient accumulation: {{grad_accum}}")
            log.info(f"Steps per epoch (with grad accum): {{steps_per_epoch}}")
            log.info(f"Completed: {{completed_epochs:.2f}} epochs ({{completed_epochs*100:.1f}}%)")
            log.info(f"Target: {{target_epochs}} epochs")
            log.info(f"Remaining: {{remaining_epochs:.2f}} epochs")
            
            # Adjust training to only do remaining epochs
            if remaining_epochs < target_epochs:
                training_args.num_train_epochs = remaining_epochs
                remaining_steps = int(remaining_epochs * steps_per_epoch)
                log.info(f"Will train for {{remaining_epochs:.2f}} epochs (~{{remaining_steps}} steps)")
            
            # Set trainer state to equivalent position
            trainer.state.global_step = equivalent_step
            trainer.state.epoch = completed_epochs
        except (ValueError, AttributeError) as e:
            log.info(f"Could not parse checkpoint name: {{e}}")

# Clear memory before training
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train
log.info("\\nStarting training...")
try:
    trainer.train(resume_from_checkpoint=trainer_resume_checkpoint)
    log.info("\\n[OK] Training completed successfully!")
except Exception as e:
    log.info(f"\\n[ERROR] Training failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save final model
log.info("\\nSaving adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
log.info(f"\\n[OK] Adapter saved to: {{OUTPUT_DIR}}")
log.info(f"Log saved to: {{log_file}}")
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    script_path.chmod(0o755)
    log.info(f"[OK] Created training script: {script_path}")
    return script_path


# ============================================================================
# Training Execution
# ============================================================================

def run_training(script_path: Path) -> bool:
    """Execute the training script."""
    try:
        # Set up environment
        env = os.environ.copy()
        
        # Add PyTorch library path for NCCL
        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists():
                ld_library_path = env.get("LD_LIBRARY_PATH", "")
                if str(torch_lib) not in ld_library_path:
                    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{ld_library_path}" if ld_library_path else str(torch_lib)
        except Exception:
            pass
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=PROJECT_ROOT,
            env=env
        )
        
        log.info("\n[OK] Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        log.error(f"Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        log.warning("Training interrupted by user")
        return False


# ============================================================================
# Cleanup Utilities
# ============================================================================

def cleanup_empty_directories(directory: Path) -> int:
    """Remove empty directories and return count."""
    removed = 0
    for item in directory.rglob('*'):
        if item.is_dir() and not any(item.iterdir()):
            try:
                item.rmdir()
                removed += 1
            except OSError:
                pass
    return removed


def cleanup_old_training_scripts(keep_recent: int = 3) -> int:
    """Remove old training scripts, keeping the most recent ones."""
    scripts = sorted(SCRIPTS_DIR.glob('finetune_*.py'), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for script in scripts[keep_recent:]:
        try:
            script.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def cleanup_old_logs(keep_recent: int = 3) -> int:
    """Remove old training logs, keeping the most recent ones per adapter."""
    removed = 0
    # Find all adapter directories
    for adapter_dir in ADAPTERS_DIR.iterdir():
        if adapter_dir.is_dir():
            # Find all log files in this adapter
            logs = sorted(adapter_dir.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
            # Remove old logs, keep most recent
            for log_file in logs[keep_recent:]:
                try:
                    log_file.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def cleanup_adapters() -> None:
    """Clean up empty adapter directories."""
    removed = cleanup_empty_directories(ADAPTERS_DIR)
    if removed > 0:
        log.info(f"  Removed {removed} empty adapter directories")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune_lora.py --model_name gpt-oss:20b
  python finetune_lora.py --model_name gpt-oss:20b --cpu_only --max_length 2048
  python finetune_lora.py --model_name gpt-oss:20b --resume_from_checkpoint checkpoint-1500
        """
    )
    
    parser.add_argument('--model_name', type=str, default='gpt-oss:20b',
                       help='Model name (Ollama format or HuggingFace ID)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace access token (or set HF_TOKEN env var)')
    parser.add_argument('--dataset_source', type=str, choices=['conversations', 'files', 'memories', 'all'],
                       default='all', help='Data source for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_name', type=str, default=None,
                       help='Output adapter name (default: lora-{model}-{timestamp})')
    parser.add_argument('--export_only', action='store_true',
                       help='Only export dataset, do not train')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Use existing dataset file')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum sequence length (default: 192 for GPU, 2048 for CPU)')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Use CPU-only training (allows longer sequences)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume from checkpoint (e.g., checkpoint-1500)')
    parser.add_argument('--no_deepspeed', action='store_true',
                       help='Disable DeepSpeed (not recommended for large models)')
    
    args = parser.parse_args()
    
    # Set default max_length
    if args.max_length is None:
        args.max_length = 2048 if args.cpu_only else 192
    
    # Generate output name
    if args.output_name is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        model_safe = args.model_name.replace(':', '-').replace('/', '-')
        args.output_name = f"lora-{model_safe}-{timestamp}"
    
    # Cleanup
    log.info("Performing initial cleanup...")
    cleanup_adapters()
    old_scripts = cleanup_old_training_scripts(keep_recent=3)
    if old_scripts > 0:
        log.info(f"  Removed {old_scripts} old training scripts")
    old_logs = cleanup_old_logs(keep_recent=3)
    if old_logs > 0:
        log.info(f"  Removed {old_logs} old log files")
    log.info("")
    
    # Print configuration
    log.info("="*70)
    log.info("LoRA Fine-Tuning")
    log.info("="*70)
    log.info(f"Model: {args.model_name}")
    log.info(f"Dataset: {args.dataset_source}")
    log.info(f"Epochs: {args.epochs}")
    log.info(f"Learning Rate: {args.learning_rate}")
    log.info(f"Max Length: {args.max_length}")
    log.info(f"Mode: {'CPU-only' if args.cpu_only else 'GPU' + (' + DeepSpeed' if not args.no_deepspeed else '')}")
    log.info(f"Output: {args.output_name}")
    log.info("="*70)
    log.info("")
    
    # Export dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            log.error(f"Dataset file not found: {dataset_path}")
            return 1
        log.info(f"Using existing dataset: {dataset_path}")
    else:
        log.info("Step 1: Exporting training data...")
        dataset_path = export_training_data(source=args.dataset_source)
        log.info("")
    
    if args.export_only:
        log.info("[OK] Dataset exported. Use --dataset_path to train later.")
        return 0
    
    # Create training script
    log.info("Step 2: Creating training script...")
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    script_path = create_training_script(
        model_name=args.model_name,
        dataset_path=dataset_path,
        output_name=args.output_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hf_token=hf_token,
        max_length=args.max_length,
        cpu_only=args.cpu_only,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_deepspeed=not args.no_deepspeed
    )
    log.info("")
    
    # Run training
    log.info("Step 3: Running training...")
    success = run_training(script_path)
    
    # Final cleanup
    log.info("\nCleaning up...")
    cleanup_adapters()
    old_scripts = cleanup_old_training_scripts(keep_recent=3)
    if old_scripts > 0:
        log.info(f"  Removed {old_scripts} old training scripts")
    old_logs = cleanup_old_logs(keep_recent=3)
    if old_logs > 0:
        log.info(f"  Removed {old_logs} old log files")
    
    if success:
        adapter_path = ADAPTERS_DIR / args.output_name
        log.info(f"\n[OK] Fine-tuning complete!")
        log.info(f"   Adapter: {adapter_path}")
        return 0
    else:
        log.error("Fine-tuning failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

