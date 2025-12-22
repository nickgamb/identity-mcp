#!/usr/bin/env python3
"""
Clean training data before fine-tuning.
Uses the same folder structure as finetune_lora.py.

Run this BEFORE training to:
1. Remove empty/too-short examples
2. Filter problematic content
3. Report stats on what was removed

Usage:
    python clean_training_data.py                    # Analyze only (dry run)
    python clean_training_data.py --fix              # Clean and rewrite files
    python clean_training_data.py --min_response 50  # Custom min response length
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Same paths as finetune_lora.py
PROJECT_ROOT = Path(__file__).parent.parent.parent if Path(__file__).parent.name == "conversation_processing" else Path(__file__).parent
USER_ID = os.environ.get("USER_ID")

def get_user_dir(base_dir: Path) -> Path:
    if USER_ID:
        return base_dir / USER_ID
    return base_dir

CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations")
FILES_DIR = get_user_dir(PROJECT_ROOT / "files")
MEMORY_DIR = get_user_dir(PROJECT_ROOT / "memory")


@dataclass
class CleaningStats:
    total: int = 0
    kept: int = 0
    empty_response: int = 0
    too_short: int = 0
    too_long: int = 0
    bad_chars: int = 0
    empty_instruction: int = 0


def has_bad_chars(text: str) -> bool:
    """Check for characters that often cause NaN loss."""
    if '\x00' in text:
        return True
    # Excessive special unicode
    weird_count = sum(1 for c in text if ord(c) > 0xFFFF)
    if weird_count > len(text) * 0.1:  # More than 10% weird chars
        return True
    return False


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace('\x00', '')
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def analyze_conversations(min_response: int, max_length: int) -> Tuple[CleaningStats, List[Dict]]:
    """Analyze conversation files for issues."""
    stats = CleaningStats()
    issues = []
    
    if not CONVERSATIONS_DIR.exists():
        print(f"  No conversations directory: {CONVERSATIONS_DIR}")
        return stats, issues
    
    jsonl_files = sorted(CONVERSATIONS_DIR.glob('conversation_*.jsonl'))
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                messages = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        messages.append((line_num, msg))
                    except json.JSONDecodeError:
                        continue
                
                # Check user/assistant pairs
                for i in range(len(messages) - 1):
                    line_num, user_msg = messages[i]
                    _, assistant_msg = messages[i + 1]
                    
                    if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                        continue
                    
                    stats.total += 1
                    instruction = user_msg.get('content', '')
                    response = assistant_msg.get('content', '')
                    
                    issue = None
                    if not response or response.strip() == '':
                        stats.empty_response += 1
                        issue = "empty_response"
                    elif not instruction or instruction.strip() == '':
                        stats.empty_instruction += 1
                        issue = "empty_instruction"
                    elif len(response) < min_response:
                        stats.too_short += 1
                        issue = f"too_short ({len(response)} chars)"
                    elif len(response) > max_length * 4:  # Rough char estimate
                        stats.too_long += 1
                        issue = f"too_long ({len(response)} chars)"
                    elif has_bad_chars(response) or has_bad_chars(instruction):
                        stats.bad_chars += 1
                        issue = "bad_chars"
                    else:
                        stats.kept += 1
                    
                    if issue:
                        issues.append({
                            'file': jsonl_file.name,
                            'line': line_num,
                            'issue': issue,
                            'preview': instruction[:50] + '...' if len(instruction) > 50 else instruction
                        })
                        
        except Exception as e:
            print(f"  Error reading {jsonl_file}: {e}")
    
    return stats, issues


def analyze_files(min_response: int, max_length: int) -> Tuple[CleaningStats, List[Dict]]:
    """Analyze markdown/text files for issues."""
    stats = CleaningStats()
    issues = []
    
    if not FILES_DIR.exists():
        print(f"  No files directory: {FILES_DIR}")
        return stats, issues
    
    for ext in ['*.txt', '*.md']:
        for file_path in FILES_DIR.rglob(ext):
            stats.total += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                issue = None
                if not content or content.strip() == '':
                    stats.empty_response += 1
                    issue = "empty"
                elif len(content) < min_response:
                    stats.too_short += 1
                    issue = f"too_short ({len(content)} chars)"
                elif len(content) > max_length * 4:
                    stats.too_long += 1
                    issue = f"too_long ({len(content)} chars)"
                elif has_bad_chars(content):
                    stats.bad_chars += 1
                    issue = "bad_chars"
                else:
                    stats.kept += 1
                
                if issue:
                    issues.append({
                        'file': str(file_path.relative_to(FILES_DIR)),
                        'issue': issue
                    })
                    
            except Exception as e:
                issues.append({'file': str(file_path), 'issue': f"read_error: {e}"})
    
    return stats, issues


def analyze_memory(min_response: int) -> Tuple[CleaningStats, List[Dict]]:
    """Analyze memory files for issues."""
    stats = CleaningStats()
    issues = []
    
    if not MEMORY_DIR.exists():
        print(f"  No memory directory: {MEMORY_DIR}")
        return stats, issues
    
    for jsonl_file in MEMORY_DIR.glob('*.jsonl'):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        stats.total += 1
                        
                        content = record.get('content', '')
                        if not isinstance(content, str):
                            content = json.dumps(content)
                        
                        issue = None
                        if not content or content.strip() == '':
                            stats.empty_response += 1
                            issue = "empty"
                        elif len(content) < 10:  # Memories can be short
                            stats.too_short += 1
                            issue = f"too_short ({len(content)} chars)"
                        elif has_bad_chars(content):
                            stats.bad_chars += 1
                            issue = "bad_chars"
                        else:
                            stats.kept += 1
                        
                        if issue:
                            issues.append({
                                'file': jsonl_file.name,
                                'line': line_num,
                                'issue': issue
                            })
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"  Error reading {jsonl_file}: {e}")
    
    return stats, issues


def print_stats(name: str, stats: CleaningStats, issues: List[Dict], verbose: bool):
    """Print analysis results."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Total examples:    {stats.total}")
    print(f"  ✅ Kept:           {stats.kept}")
    print(f"  ❌ Empty response: {stats.empty_response}")
    print(f"  ❌ Empty instruct: {stats.empty_instruction}")
    print(f"  ❌ Too short:      {stats.too_short}")
    print(f"  ❌ Too long:       {stats.too_long}")
    print(f"  ❌ Bad chars:      {stats.bad_chars}")
    
    if verbose and issues:
        print(f"\n  First 10 issues:")
        for issue in issues[:10]:
            print(f"    - {issue}")


def fix_conversations(min_response: int, max_length: int):
    """Rewrite conversation files, removing bad examples."""
    if not CONVERSATIONS_DIR.exists():
        return
    
    jsonl_files = sorted(CONVERSATIONS_DIR.glob('conversation_*.jsonl'))
    total_removed = 0
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            cleaned_lines = []
            removed = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    content = msg.get('content', '')
                    
                    # Clean the content
                    if isinstance(content, str):
                        content = clean_text(content)
                        msg['content'] = content
                    
                    # Skip if empty or bad
                    if not content or has_bad_chars(content):
                        removed += 1
                        continue
                    
                    cleaned_lines.append(json.dumps(msg, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    removed += 1
                    continue
            
            if removed > 0:
                # Backup original
                backup_path = jsonl_file.with_suffix('.jsonl.bak')
                jsonl_file.rename(backup_path)
                
                # Write cleaned
                with open(jsonl_file, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                
                print(f"  Cleaned {jsonl_file.name}: removed {removed} messages")
                total_removed += removed
                
        except Exception as e:
            print(f"  Error fixing {jsonl_file}: {e}")
    
    print(f"\n  Total messages removed from conversations: {total_removed}")


def main():
    parser = argparse.ArgumentParser(description='Clean training data for fine-tuning')
    parser.add_argument('--fix', action='store_true', help='Actually clean files (default: analyze only)')
    parser.add_argument('--min_response', type=int, default=20, help='Minimum response length in chars')
    parser.add_argument('--max_length', type=int, default=2048, help='Max length in tokens (chars * 4 estimate)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show individual issues')
    args = parser.parse_args()
    
    print("\n📊 Analyzing training data...")
    print(f"   Conversations: {CONVERSATIONS_DIR}")
    print(f"   Files: {FILES_DIR}")
    print(f"   Memory: {MEMORY_DIR}")
    
    # Analyze each source
    conv_stats, conv_issues = analyze_conversations(args.min_response, args.max_length)
    print_stats("Conversations", conv_stats, conv_issues, args.verbose)
    
    file_stats, file_issues = analyze_files(args.min_response, args.max_length)
    print_stats("Files", file_stats, file_issues, args.verbose)
    
    mem_stats, mem_issues = analyze_memory(args.min_response)
    print_stats("Memory", mem_stats, mem_issues, args.verbose)
    
    # Summary
    total = conv_stats.total + file_stats.total + mem_stats.total
    kept = conv_stats.kept + file_stats.kept + mem_stats.kept
    bad = total - kept
    
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Total examples: {total}")
    print(f"  ✅ Clean: {kept} ({100*kept/max(1,total):.1f}%)")
    print(f"  ❌ Problematic: {bad} ({100*bad/max(1,total):.1f}%)")
    
    if args.fix:
        print(f"\n🔧 Fixing files...")
        fix_conversations(args.min_response, args.max_length)
        print("\n✅ Done. Original files backed up with .bak extension")
    else:
        if bad > 0:
            print(f"\n💡 Run with --fix to clean the data")
            print(f"   python clean_training_data.py --fix")


if __name__ == "__main__":
    main()