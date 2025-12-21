#!/usr/bin/env python3
"""
Conversation Parser - Convert ChatGPT Export to JSONL/Markdown
==============================================================

This script reads your raw conversations.json export from ChatGPT and converts
each conversation into clean, structured files:
  - conversation_*.jsonl - One JSON object per line (message)
  - conversation_*.md - Human-readable markdown format

This is typically the FIRST script you run after exporting your ChatGPT data.

Usage:
  python parse_conversations.py                    # Process all conversations
  python parse_conversations.py --workers 4        # Use 4 parallel workers
  python parse_conversations.py --clean            # Remove existing output first
  python parse_conversations.py --input path.json  # Custom input file

Input:
  conversations/conversations.json (default)
  
Output:
  conversations/conversation_<id>.jsonl
  conversations/conversation_<id>.md

The JSONL format contains one message per line:
  {"timestamp": "...", "role": "user|assistant", "content": "...", "line_number": 1}

Dependencies:
  pip install tqdm  (for progress bars)
"""

import os
import json
import argparse
import random
import string
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, Lock, cpu_count
from typing import List, Dict, Any, Optional, Set
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Support multi-user: read USER_ID from environment
USER_ID = os.environ.get("USER_ID")
def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    """Get user-specific directory if user_id is provided, otherwise base directory."""
    if user_id:
        return base_dir / user_id
    return base_dir

CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations", USER_ID)

# Global lock for file operations
file_lock = Lock()


def generate_unique_id() -> str:
    """Generate a unique ID using timestamp and random suffix."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{timestamp}_{random_suffix}"


def extract_messages_from_mapping(mapping: Dict, worker_id: int = 0, show_progress: bool = False) -> List[Dict]:
    """
    Extract messages from ChatGPT's nested mapping structure.
    
    ChatGPT exports use a tree structure where each node can have a message.
    This extracts all valid user/assistant messages.
    """
    messages = []
    total_nodes = len(mapping)
    
    # Create iterator with optional progress bar
    if show_progress and HAS_TQDM:
        iterator = tqdm(
            enumerate(mapping.items()),
            desc=f"Worker {worker_id:3d}",
            total=total_nodes,
            leave=False,
            position=worker_id % 4 + 1,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
            unit="msg"
        )
    else:
        iterator = enumerate(mapping.items())
    
    for i, (node_id, node) in iterator:
        message = node.get('message')
        if not message:
            continue
            
        author = message.get('author')
        content = message.get('content')
        
        if not author or not content:
            continue
            
        role = author.get('role', 'unknown')
        if role not in ('user', 'assistant'):
            continue
            
        parts = content.get('parts', [])
        if not parts:
            continue
            
        # Get the text content
        text_content = parts[0] if isinstance(parts[0], str) else json.dumps(parts[0])
        if not text_content or not text_content.strip():
            continue
        
        # Clean invalid UTF-8 sequences early
        if isinstance(text_content, str):
            text_content = text_content.encode('utf-8', errors='replace').decode('utf-8')
            text_content = text_content.replace('\x00', '')  # Remove null bytes
            
        # Get timestamp
        create_time = message.get('create_time')
        if create_time:
            try:
                timestamp = datetime.fromtimestamp(create_time).isoformat()
            except (ValueError, OSError):
                timestamp = "Unknown"
        else:
            timestamp = "Unknown"
        
        messages.append({
            'timestamp': timestamp,
            'role': role,
            'content': text_content,
            'line_number': len(messages) + 1
        })
    
    return messages


def process_conversation(args) -> Optional[Dict]:
    """
    Process a single conversation into JSONL and Markdown files.
    
    Args:
        args: Tuple of (conversation dict, output directory, worker id, show_progress)
    
    Returns:
        Dict with processing results or None if failed
    """
    convo, output_dir, worker_id, show_progress = args
    
    # Get conversation ID
    convo_id = convo.get('id', '')
    if not convo_id:
        convo_id = convo.get('conversation_id', generate_unique_id())
    
    # Get title for markdown header
    title = convo.get('title', 'Untitled Conversation')
    
    # Extract messages from mapping
    mapping = convo.get('mapping', {})
    if not mapping:
        return None
    
    messages = extract_messages_from_mapping(mapping, worker_id, show_progress)
    if not messages:
        return None
    
    # Sort by timestamp
    def parse_timestamp(msg):
        ts = msg.get('timestamp', '')
        if ts and ts != 'Unknown':
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return datetime.min
    
    messages.sort(key=parse_timestamp)
    
    # Update line numbers after sorting
    for i, msg in enumerate(messages):
        msg['line_number'] = i + 1
    
    # Output paths
    filename_base = f"conversation_{convo_id}"
    jsonl_path = output_dir / f"{filename_base}.jsonl"
    md_path = output_dir / f"{filename_base}.md"
    
    # Write files with lock to avoid conflicts
    with file_lock:
        # Write JSONL
        with open(jsonl_path, 'w', encoding='utf-8', errors='replace') as f:
            for msg in messages:
                # Clean message content to ensure valid UTF-8 and JSON
                cleaned_msg = {}
                for key, value in msg.items():
                    if isinstance(value, str):
                        # Clean invalid UTF-8 sequences
                        cleaned = value.encode('utf-8', errors='replace').decode('utf-8')
                        # Remove null bytes and replacement characters that might break JSON
                        cleaned = cleaned.replace('\x00', '').replace('\ufffd', '')
                        cleaned_msg[key] = cleaned
                    else:
                        cleaned_msg[key] = value
                
                # Validate JSON before writing
                try:
                    json_str = json.dumps(cleaned_msg, ensure_ascii=False)
                    json.loads(json_str)  # Validate it can be parsed
                    f.write(json_str + '\n')
                except Exception as e:
                    print(f"Warning: Skipping invalid message in {convo_id}: {e}")
                    continue
        
        # Write Markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Conversation ID:** {convo_id}\n\n")
            f.write(f"**Messages:** {len(messages)}\n\n")
            f.write("---\n\n")
            
            for msg in messages:
                role_display = msg['role'].title()
                timestamp = msg['timestamp']
                content = msg['content']
                
                f.write(f"**{role_display}** [{timestamp}]:\n\n")
                f.write(f"{content}\n\n")
                f.write("---\n\n")
    
    return {
        'id': convo_id,
        'title': title,
        'message_count': len(messages),
        'jsonl_path': str(jsonl_path),
        'md_path': str(md_path)
    }


def load_conversations(input_path: Path) -> List[Dict]:
    """Load conversations from JSON file."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading conversations from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            print(f"Error: Unexpected data format in {input_path}")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}")
        sys.exit(1)


def clean_output_directory(output_dir: Path) -> int:
    """Remove existing conversation files from output directory."""
    removed = 0
    
    for pattern in ['conversation_*.jsonl', 'conversation_*.md']:
        for file_path in output_dir.glob(pattern):
            try:
                file_path.unlink()
                removed += 1
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    return removed


def get_existing_conversation_ids(output_dir: Path) -> Set[str]:
    """Get IDs of conversations that have already been processed."""
    existing = set()
    
    for jsonl_file in output_dir.glob('conversation_*.jsonl'):
        # Extract ID from filename: conversation_<id>.jsonl
        name = jsonl_file.stem  # conversation_<id>
        if name.startswith('conversation_'):
            conv_id = name[len('conversation_'):]
            existing.add(conv_id)
    
    return existing


def main():
    parser = argparse.ArgumentParser(
        description='Convert ChatGPT conversations.json to JSONL and Markdown files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to conversations.json (default: conversations/conversations.json)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (default: conversations/)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count, max {cpu_count()})')
    parser.add_argument('--clean', action='store_true',
                        help='Remove existing conversation files before processing')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Reprocess all conversations, even if already processed')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-worker progress bars (requires tqdm)')
    
    args = parser.parse_args()
    
    # Set paths
    input_path = Path(args.input) if args.input else CONVERSATIONS_DIR / 'conversations.json'
    output_dir = Path(args.output) if args.output else CONVERSATIONS_DIR
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean if requested
    if args.clean:
        removed = clean_output_directory(output_dir)
        if not args.quiet:
            print(f"Removed {removed} existing conversation files")
    
    # Load conversations
    conversations = load_conversations(input_path)
    
    if not conversations:
        print("No conversations found in input file")
        return
    
    if not args.quiet:
        print(f"Found {len(conversations)} conversations")
    
    # Filter out already processed conversations (unless --force)
    if not args.force and not args.clean:
        existing_ids = get_existing_conversation_ids(output_dir)
        if existing_ids:
            original_count = len(conversations)
            conversations = [c for c in conversations if c.get('id', '') not in existing_ids]
            skipped = original_count - len(conversations)
            if skipped > 0 and not args.quiet:
                print(f"Skipping {skipped} already processed conversations (use --force to reprocess)")
    
    if not conversations:
        print("All conversations already processed. Use --force to reprocess.")
        return
    
    # Set up workers
    num_workers = args.workers if args.workers else min(cpu_count(), len(conversations))
    num_workers = max(1, min(num_workers, len(conversations)))
    
    if not args.quiet:
        print(f"Processing {len(conversations)} conversations using {num_workers} worker(s)...")
    
    # Process conversations
    results = []
    show_worker_progress = args.verbose and HAS_TQDM and not args.quiet
    work_items = [(convo, output_dir, i % num_workers, show_worker_progress) 
                  for i, convo in enumerate(conversations)]
    
    if num_workers == 1:
        # Single-threaded processing
        if HAS_TQDM and not args.quiet:
            pbar = tqdm(
                total=len(conversations),
                desc="Processing",
                unit="convo",
                position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            for item in work_items:
                result = process_conversation(item)
                if result:
                    results.append(result)
                pbar.update(1)
            pbar.close()
        else:
            for i, item in enumerate(work_items):
                result = process_conversation(item)
                if result:
                    results.append(result)
                if not args.quiet and (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(conversations)}...")
    else:
        # Parallel processing with overall progress bar
        if HAS_TQDM and not args.quiet:
            with tqdm(
                total=len(conversations),
                desc="Overall Progress",
                unit="convo",
                position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                with Pool(num_workers) as pool:
                    for result in pool.imap_unordered(process_conversation, work_items):
                        if result:
                            results.append(result)
                        pbar.update(1)
        else:
            with Pool(num_workers) as pool:
                for i, result in enumerate(pool.imap_unordered(process_conversation, work_items)):
                    if result:
                        results.append(result)
                    if not args.quiet and (i + 1) % 50 == 0:
                        print(f"  Processed {i + 1}/{len(conversations)}...")
    
    # Summary
    total_messages = sum(r['message_count'] for r in results)
    
    print(f"\n✅ Processed {len(results)} conversations")
    print(f"   Total messages: {total_messages}")
    print(f"   Output directory: {output_dir}")
    
    if results:
        # Show some stats
        avg_messages = total_messages / len(results)
        print(f"   Average messages per conversation: {avg_messages:.1f}")


if __name__ == '__main__':
    main()

