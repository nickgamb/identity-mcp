#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LoRA Fine-Tuning for Large Language Models                 ║
║                                                                               ║
║  Supports models up to 20B+ parameters on consumer/prosumer hardware          ║
║  • CPU-only mode (slow but always works, ~40GB RAM for 20B model)             ║
║  • CPU with GPU offload (loads on CPU, offloads shards to GPU during training)║
║  • Checkpoint save/resume with full state preservation                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

source venv/bin/activate
watch -n 1 nvidia-smi

CPU only with defaults
python scripts/conversation_processing/finetune_lora.py --model_name gpt-oss:20b --max_length 2048 --epochs 3

CPU + GPU offload with defaults

python scripts/conversation_processing/finetune_lora.py --model_name gpt-oss:20b --gpu_offload --max_length 2048 --epochs 3

CPU + GPU offload with custom settings
python scripts/conversation_processing/finetune_lora.py --model_name gpt-oss:20b --gpu_offload --max_length 2048 --epochs 3 --gpu_max_memory_gb 12 --gpu_free_fraction 0.6 --reenable_gpu_every 2

python scripts/conversation_processing/finetune_lora.py --model_name gpt-oss:20b --gpu_offload --max_length 2048 --epochs 3 --gpu_max_memory_gb 14 --gpu_free_fraction 0.7 --reenable_gpu_every 2

python scripts/conversation_processing/finetune_lora.py --model_name gpt-oss:20b --gpu_offload --max_length 2048 --epochs 3 --gpu_max_memory_gb 16 --gpu_free_fraction 0.7 --reenable_gpu_every 2
"""

import os
import re
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
    model_name: str = "gpt-oss:20b"
    dataset_source: str = "all"
    epochs: int = 3
    learning_rate: float = 1e-5
    max_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    checkpoint_steps: int = 30
    logging_steps: int = 50
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    use_gpu_offload: bool = False
    gpu_max_memory_gb: int = 8          # hard cap per GPU used for "support only" sharding
    gpu_free_fraction: float = 0.50     # use at most this fraction of *currently free* VRAM
    reenable_gpu_every: int = 200       # after CPU fallback, try re-enable GPU offload every N optimizer steps


PROJECT_ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.name == "conversation_processing" else Path(__file__).parent
USER_ID = os.environ.get("USER_ID")

def get_user_dir(base_dir: Path) -> Path:
    if USER_ID:
        return base_dir / USER_ID
    return base_dir

CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations")
FILES_DIR = get_user_dir(PROJECT_ROOT / "files")
MEMORY_DIR = get_user_dir(PROJECT_ROOT / "memory")
TRAINING_DATA_DIR = get_user_dir(PROJECT_ROOT / "training_data")
ADAPTERS_DIR = get_user_dir(PROJECT_ROOT / "adapters")

for d in [TRAINING_DATA_DIR, ADAPTERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        if record.levelname == 'INFO':
            return f"{color}{record.getMessage()}{self.RESET}"
        return f"{color}{self.BOLD}[{record.levelname}]{self.RESET} {color}{record.getMessage()}{self.RESET}"


def setup_logging(output_dir: Optional[Path] = None, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("finetune")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

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
    width = 70
    print(f"\n\033[1m{char * width}\033[0m")
    print(f"\033[1m  {title.center(width - 4)}\033[0m")
    print(f"\033[1m{char * width}\033[0m\n")


def print_config_table(config: dict):
    max_key = max(len(str(k)) for k in config.keys())
    for key, value in config.items():
        print(f"  \033[36m{key:<{max_key}}\033[0m : \033[33m{value}\033[0m")
    print()


def get_gpu_info() -> dict:
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
                "has_tensor_cores": props.major >= 7,
            })

        return {"available": True, "count": len(devices), "devices": devices}
    except Exception as e:
        return {"available": False, "count": 0, "devices": [], "error": str(e)}


def get_model_id(model_name: str) -> str:
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


def _move_batch_to_device(batch: dict, device):
    out = {}
    for k, v in batch.items():
        try:
            out[k] = v.to(device, non_blocking=True)
        except Exception:
            out[k] = v
    return out


def _force_model_dtype_inplace(model, target_dtype):
    # Force a consistent dtype across the entire sharded model.
    # This avoids float != bf16/fp16 mixed errors.
    import torch
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != target_dtype:
                p.data = p.data.to(dtype=target_dtype)
        for b in model.buffers():
            if b.is_floating_point() and b.dtype != target_dtype:
                b.data = b.data.to(dtype=target_dtype)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_conversations() -> List[Dict[str, Any]]:
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
                            messages.append({'role': msg.get('role'), 'content': msg.get('content', '')})
                    except json.JSONDecodeError:
                        continue

                if messages:
                    conversations.append({'id': jsonl_file.stem.replace('conversation_', ''), 'messages': messages})
        except Exception as e:
            logging.warning(f"Error loading {jsonl_file}: {e}")

    return conversations


def load_files() -> List[Dict[str, Any]]:
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

                files.append({'filename': file_path.name, 'title': title, 'content': content})
            except Exception as e:
                logging.warning(f"Error loading {file_path}: {e}")

    return files


def load_memory() -> List[Dict[str, Any]]:
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
    if output_path is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        output_path = TRAINING_DATA_DIR / f"dataset-{timestamp}.jsonl"

    training_examples = []

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
            training_examples.append({'instruction': instruction, 'response': content})
        print(f"  🧠 Loaded {len(training_examples) - start_count} examples from memory")

    random.shuffle(training_examples)

    valid_count = 0
    with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
        for example in training_examples:
            try:
                cleaned = {}
                for key, value in example.items():
                    if isinstance(value, str):
                        cleaned[key] = value.encode('utf-8', errors='replace').decode('utf-8')
                        cleaned[key] = cleaned[key].replace('\x00', '').replace('\ufffd', '')
                    else:
                        cleaned[key] = value

                json_str = json.dumps(cleaned, ensure_ascii=False)
                json.loads(json_str)
                f.write(json_str + '\n')
                valid_count += 1
            except Exception:
                continue

    print(f"  ✅ Wrote {valid_count} examples to {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    config: Config,
    resume_from_checkpoint: Optional[str] = None,
    log: Optional[logging.Logger] = None,
):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if log is None:
        log = logging.getLogger("finetune")

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    use_gpu = config.use_gpu_offload and torch.cuda.is_available()

    if use_gpu:
        gpu_info = get_gpu_info()
        log.info("🖥️  CPU-first training with GPU offload")
        log.info("   GPU is support only (hard capped per GPU)")
        if gpu_info.get("available"):
            for i, dev in enumerate(gpu_info.get("devices", [])):
                log.info(f"   GPU [{i}]: {dev['name']} ({dev['memory_total_gb']}GB)")
        log.info("")
    else:
        log.info("🖥️  CPU-only training mode")
        log.info("")

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    log.info("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_info = get_gpu_info()
    num_gpus = gpu_info["count"] if use_gpu else 0

    # Choose ONE dtype for the whole model to avoid float/bf16 mixed errors.
    # - GPU-offload path: use FP16 everywhere (CPU shards + GPU shards) for consistency.
    # - CPU-only path: use FP32 everywhere (safer on CPU).
    target_dtype = torch.bfloat16 if (use_gpu and num_gpus > 0) else torch.float32

    # Always load base model on CPU first (prevents GPU-side load/dequant OOM)
    log.info("📦 Loading model to CPU (this takes a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=target_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = _force_model_dtype_inplace(model, target_dtype)
    log.info("   ✅ Model loaded on CPU!")

    # Checkpoint handling
    global_step = 0
    saved_optimizer_state = None
    saved_scheduler_state = None

    checkpoint_path = None
    if resume_from_checkpoint:
        cp = Path(resume_from_checkpoint)
        if not cp.is_absolute():
            cp = output_dir / resume_from_checkpoint
        if cp.exists():
            checkpoint_path = cp

    if not checkpoint_path:
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoints.sort(key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0)
            checkpoint_path = checkpoints[-1]

    if checkpoint_path and checkpoint_path.exists():
        log.info(f"📂 Loading checkpoint: {checkpoint_path}")
        # Suppress PEFT's "missing adapter keys" warning - it's a false positive
        # when loading safetensors checkpoints with correct key names
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*missing adapter keys.*", category=UserWarning)
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        log.info("   ✅ Checkpoint weights loaded (ignore any 'missing adapter keys' warning above)")

        if "checkpoint-" in checkpoint_path.name:
            try:
                global_step = int(checkpoint_path.name.split("checkpoint-")[1])
            except Exception:
                pass

        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            log.info("   Loading optimizer/scheduler state...")
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            saved_optimizer_state = state.get("optimizer")
            saved_scheduler_state = state.get("scheduler")
            if "global_step" in state:
                global_step = state["global_step"]

        log.info(f"   Resuming from step {global_step}")
    else:
        log.info("🔧 Applying LoRA adapters...")
        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    # Ensure dtype consistency after PEFT wraps modules
    model = _force_model_dtype_inplace(model, target_dtype)

    # Reduce VRAM pressure
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass

    # REQUIRED for gradient checkpointing + PEFT
    # (fixes: "None of the inputs have requires_grad=True. Gradients will be None")
    try:
        model.enable_input_require_grads()
    except Exception:
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                emb.weight.requires_grad_(True)
        except Exception:
            pass

    model.print_trainable_parameters()

    # ---- Accelerate dispatch helpers ----
    def _compute_max_memory():
        mm = {"cpu": "100GiB"}
        if not (torch.cuda.is_available() and num_gpus > 0):
            return mm

        cap = float(config.gpu_max_memory_gb)
        frac = float(config.gpu_free_fraction)

        for gi in range(num_gpus):
            try:
                free_b, total_b = torch.cuda.mem_get_info(gi)
                free_gb = free_b / (1024**3)
                total_gb = total_b / (1024**3)

                allow = min(cap, free_gb * frac)
                allow = max(2.0, allow)
                allow = min(allow, max(2.0, total_gb - 3.0))  # headroom for activations/fragmentation

                mm[gi] = f"{allow:.0f}GiB"
            except Exception:
                mm[gi] = "4GiB"
        return mm

    def dispatch_cpu_only(m):
        # Properly remove GPU sharding without manual param moves.
        try:
            from accelerate import dispatch_model
            from accelerate.hooks import remove_hook_from_module
        except Exception:
            m = m.to("cpu")
            return _force_model_dtype_inplace(m, target_dtype)

        try:
            try:
                remove_hook_from_module(m, recurse=True)
            except Exception:
                pass
            m = dispatch_model(m, device_map={"": "cpu"})
            m.to("cpu")
        except Exception:
            m = m.to("cpu")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return _force_model_dtype_inplace(m, target_dtype)

    def try_dispatch_to_gpus(m):
        if not (config.use_gpu_offload and torch.cuda.is_available() and num_gpus > 0):
            return None
        try:
            from accelerate import dispatch_model
            from accelerate.utils import infer_auto_device_map
            from accelerate.hooks import remove_hook_from_module
        except Exception:
            return None

        budgets = _compute_max_memory()
        try:
            try:
                remove_hook_from_module(m, recurse=True)
            except Exception:
                pass

            device_map = infer_auto_device_map(
                m,
                max_memory=budgets,
                no_split_module_classes=getattr(m, "_no_split_modules", None),
            )
            m = dispatch_model(m, device_map=device_map)
            return _force_model_dtype_inplace(m, target_dtype)
        except Exception:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            return None

    model_container = [model]
    gpu_enabled = [False]

    if use_gpu and num_gpus > 0:
        # Helps fragmentation, but doesn't "force" more model onto GPU.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

        resh = try_dispatch_to_gpus(model_container[0])
        if resh is not None:
            model_container[0] = resh
            gpu_enabled[0] = True
            log.info(f"🚀 GPU offload enabled (capped ~{float(config.gpu_max_memory_gb):.0f}GiB/GPU).")
        else:
            log.warning("⚠️  GPU offload not possible right now — staying CPU-only.")
            model_container[0] = dispatch_cpu_only(model_container[0])
            gpu_enabled[0] = False

    # Dataset
    log.info(f"📊 Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def tokenize_fn(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for inst, resp in zip(examples["instruction"], examples["response"]):
            # Use the model's chat template (harmony format for gpt-oss)
            messages = [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": resp}
            ]
            
            # Apply chat template - this handles harmony format automatically
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Tokenize
            tok = tokenizer(
                text, 
                truncation=True, 
                max_length=config.max_length, 
                padding="max_length",
                return_tensors=None
            )
            
            all_input_ids.append(tok["input_ids"])
            all_attention_mask.append(tok["attention_mask"])
            all_labels.append(tok["input_ids"].copy())
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }

    log.info("   Tokenizing...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=(global_step == 0),
        num_workers=0,
    )

    # Optimizer/scheduler
    log.info("⚙️  Setting up optimizer...")
    trainable_params = [p for p in model_container[0].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=0.01)

    total_steps = len(dataloader) * config.epochs
    warmup_steps = min(config.warmup_steps, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    if saved_optimizer_state:
        optimizer.load_state_dict(saved_optimizer_state)
    if saved_scheduler_state:
        scheduler.load_state_dict(saved_scheduler_state)
    elif global_step > 0:
        for _ in range(global_step):
            scheduler.step()

    grad_accum = max(1, int(config.gradient_accumulation_steps))
    
    steps_per_epoch = len(dataloader)
    # global_step counts optimizer steps, but we need to skip batches
    # Each optimizer step = grad_accum batches
    batches_completed = global_step * grad_accum
    start_epoch = batches_completed // steps_per_epoch
    start_step_in_epoch = batches_completed % steps_per_epoch

    def _input_device(m):
        # For sharded models, this is the right anchor for where input_ids must live.
        try:
            emb = m.get_input_embeddings()
            return emb.weight.device
        except Exception:
            try:
                return next(m.parameters()).device
            except Exception:
                return torch.device("cpu")

    def process_batch_with_fallback(batch, grad_accum_steps):
        m = model_container[0]
        m = _force_model_dtype_inplace(m, target_dtype)
        dev = _input_device(m)
        batch_dev = _move_batch_to_device(batch, dev)

        try:
            outputs = m(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                labels=batch_dev["labels"],
            )
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                return None  # Signal bad batch
            (loss / grad_accum_steps).backward()
            return loss

        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg) and torch.cuda.is_available():
                log.warning("GPU OOM — re-dispatching to CPU and retrying this batch on CPU.")
                model_container[0] = dispatch_cpu_only(model_container[0])
                gpu_enabled[0] = False

                m2 = model_container[0]
                m2 = _force_model_dtype_inplace(m2, target_dtype)
                batch_cpu = _move_batch_to_device(batch, torch.device("cpu"))
                outputs = m2(
                    input_ids=batch_cpu["input_ids"],
                    attention_mask=batch_cpu["attention_mask"],
                    labels=batch_cpu["labels"],
                )
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    return None
                (loss / grad_accum_steps).backward()
                return loss
            raise

    # Training loop
    log.info("🚀 Starting training...")
    model_container[0].train()
    output_dir.mkdir(parents=True, exist_ok=True)

    micro = 0

    for epoch in range(start_epoch, config.epochs):
        log.info(f"\n{'═' * 50}")
        log.info(f"  Epoch {epoch + 1}/{config.epochs}")
        log.info(f"{'═' * 50}")

        if epoch > start_epoch:
            dataloader = DataLoader(tokenized, batch_size=config.batch_size, shuffle=True, num_workers=0)

        it = enumerate(dataloader)
        skip_batches = start_step_in_epoch if epoch == start_epoch else 0
        if skip_batches > 0:
            for _ in range(skip_batches):
                next(it, None)

        progress = tqdm(total=steps_per_epoch, initial=skip_batches,
                        desc="  Training", unit="batch")

        epoch_loss = 0.0
        nloss = 0

        for batch_idx, batch in it:
            if micro == 0:
                optimizer.zero_grad(set_to_none=True)

            loss = process_batch_with_fallback(batch, grad_accum)
            
            if loss is None:
                log.warning(f"NaN loss at batch {batch_idx}, skipping")
                continue
            
            epoch_loss += float(loss.item())
            nloss += 1
            micro += 1

            if micro >= grad_accum:
                m = model_container[0]
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                micro = 0
                global_step += 1

                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

                if global_step % config.checkpoint_steps == 0:
                    _save_checkpoint(m, tokenizer, optimizer, scheduler, global_step, epoch, epoch_loss, output_dir, log)

                # Conservative re-enable of GPU offload once VRAM is available again
                if (not gpu_enabled[0]) and use_gpu and num_gpus > 0 and (global_step % int(config.reenable_gpu_every) == 0):
                    resh = try_dispatch_to_gpus(model_container[0])
                    if resh is not None:
                        model_container[0] = resh
                        gpu_enabled[0] = True
                        log.info("✅ Re-enabled GPU offload (still capped).")

            progress.update(1)

        progress.close()
        avg = (epoch_loss / max(1, nloss))
        log.info(f"\n   Epoch {epoch + 1} complete. Average loss: {avg:.4f}")

        start_step_in_epoch = 0

    log.info(f"\n💾 Saving final model to {output_dir}...")
    try:
        model_container[0].save_pretrained(output_dir)
    except (NotImplementedError, RuntimeError, Exception):
        from safetensors.torch import save_file
        import torch
        
        state = {}
        for name, param in model_container[0].named_parameters():
            if param.requires_grad and "lora_" in name:
                if param.device.type == "meta":
                    continue
                # Keep full name - PEFT expects base_model.model.* prefix
                state[name] = param.detach().cpu().contiguous()
        
        if state:
            save_file(state, output_dir / "adapter_model.safetensors")
            _save_peft_config(model_container[0], output_dir / "adapter_config.json")
    tokenizer.save_pretrained(output_dir)
    log.info("✅ Training complete!")


def _save_peft_config(model, save_path):
    """Save PEFT config with proper JSON serialization."""
    if not hasattr(model, 'peft_config'):
        return
    for cfg in model.peft_config.values():
        config_dict = cfg.to_dict()
        # Convert non-JSON-serializable types
        cleaned = {}
        for k, v in config_dict.items():
            if isinstance(v, set):
                cleaned[k] = list(v)
            elif hasattr(v, '__name__'):  # functions/classes
                cleaned[k] = str(v)
            elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                cleaned[k] = str(v)
            else:
                cleaned[k] = v
        with open(save_path, "w") as f:
            json.dump(cleaned, f, indent=2)
        break  # Only save first config


def _save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch, epoch_loss, output_dir, log):
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            model.save_pretrained(ckpt_dir)
        except (NotImplementedError, RuntimeError, Exception) as inner_e:
            # Handle meta tensors from accelerate dispatch
            # Extract LoRA weights directly - they should be on GPU/CPU, not meta
            from safetensors.torch import save_file
            import torch
            
            state = {}
            for name, param in model.named_parameters():
                if param.requires_grad and "lora_" in name:
                    # Skip meta tensors
                    if param.device.type == "meta":
                        continue
                    # Keep full name - PEFT expects base_model.model.* prefix
                    state[name] = param.detach().cpu().contiguous()
            
            if state:
                save_file(state, ckpt_dir / "adapter_model.safetensors")
                _save_peft_config(model, ckpt_dir / "adapter_config.json")
            else:
                raise RuntimeError("No LoRA weights found to save")
        
        tokenizer.save_pretrained(ckpt_dir)
    except Exception as e:
        log.error(f"⚠️  Checkpoint save failed at step {global_step}: {e}")
        log.error("   Training will continue - you may want to stop and investigate.")
        return

    import torch
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'epoch_loss': epoch_loss,
    }, ckpt_dir / 'training_state.pt')

    log.info(f"\n   💾 Checkpoint saved: {ckpt_dir.name}")


def warn_if_other_gpu_processes():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            text=True
        ).strip()
        if out:
            print("\n⚠️  Other GPU processes detected:")
            print(out)
            print("⚠️  Consider stopping them to avoid OOM.\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--model_name', type=str, default='gpt-oss:20b')
    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--dataset_source', type=str,
                        choices=['conversations', 'files', 'memories', 'all'],
                        default='all')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--export_only', action='store_true')

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=16)

    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)

    parser.add_argument('--gpu_offload', action='store_true')

    parser.add_argument('--output_name', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    parser.add_argument("--gpu_max_memory_gb", type=int, default=8,
                        help="Max VRAM (GiB) per GPU used for offload")
    parser.add_argument("--gpu_free_fraction", type=float, default=0.50,
                        help="Use at most this fraction of currently FREE VRAM for offload budgets")
    parser.add_argument("--reenable_gpu_every", type=int, default=200,
                        help="After a fallback to CPU, try re-enabling GPU offload every N optimizer steps")

    args = parser.parse_args()

    # If CPU-only, disable CUDA BEFORE anything imports torch/cuda.
    if not args.gpu_offload:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.output_name is None:
        timestamp = int(datetime.now().timestamp() * 1000)
        model_safe = args.model_name.replace(':', '-').replace('/', '-')
        args.output_name = f"lora-{model_safe}-{timestamp}"

    output_dir = ADAPTERS_DIR / args.output_name
    log = setup_logging(output_dir, rank=0)

    print_banner("LoRA Fine-Tuning")

    print("\033[1m📋 Configuration:\033[0m")
    mode_str = "CPU with GPU offload" if args.gpu_offload else "CPU-only"
    print_config_table({
        "Model": args.model_name,
        "Dataset": args.dataset_source,
        "Epochs": args.epochs,
        "Learning Rate": args.learning_rate,
        "Max Length": args.max_length,
        "Mode": mode_str,
        "Output": args.output_name,
        "GPU cap": f"{args.gpu_max_memory_gb:.0f}GiB/GPU" if args.gpu_offload else "n/a",
        "GPU frac": f"{args.gpu_free_fraction:.2f}" if args.gpu_offload else "n/a",
        "GPU retry": f"every {args.reenable_gpu_every} steps" if args.gpu_offload else "n/a",
    })

    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        print(f"\033[1m🎮 GPUs:\033[0m")
        for i, dev in enumerate(gpu_info["devices"]):
            print(f"  [{i}] {dev['name']} ({dev['memory_total_gb']}GB)")
        print()
    else:
        if args.gpu_offload:
            log.warning("⚠️  GPU offload requested but no GPUs available - falling back to CPU-only")
        print("\033[1m🖥️  No GPUs available - will use CPU\033[0m\n")

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
        use_gpu_offload=args.gpu_offload,
        gpu_max_memory_gb=args.gpu_max_memory_gb,
        gpu_free_fraction=args.gpu_free_fraction,
        reenable_gpu_every=args.reenable_gpu_every,
    )

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

    model_id = get_model_id(args.model_name)

    warn_if_other_gpu_processes()

    train(
        model_id=model_id,
        dataset_path=dataset_path,
        output_dir=output_dir,
        config=config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        log=log,
    )

    print_banner("Training Complete!", char="═")
    log.info(f"🎉 Adapter saved to: {output_dir}")
    log.info(f"  • Test: ollama run {args.model_name} --lora {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())