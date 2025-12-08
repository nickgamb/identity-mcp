#!/usr/bin/env python3
"""
================================================================================
IDENTITY ANALYZER - Conversational Identity Pattern Analysis
================================================================================

PURPOSE:
    Analyzes parsed conversations for identity patterns without requiring
    domain-specific configuration. Uses statistical analysis to discover
    relational, stylistic, and self-referential identity markers.

PREREQUISITES:
    - Run parse_conversations.py first to generate conversation_*.jsonl files
    - Run analyze_patterns.py to generate patterns.jsonl (optional)

INPUT:
    - conversations/*.jsonl (parsed conversation files)
    - memory/patterns.jsonl (discovered patterns, optional)

OUTPUT:
    - memory/identity_analysis.jsonl (identity pattern data)
    - conversations/identity_report.md (human-readable analysis)

CONCEPTS ANALYZED:

1. RELATIONAL IDENTITY
   - Co-occurrence clusters (terms that appear together)
   - First-person plural vs singular usage ("we" vs "I")
   - Role/relationship terminology
   - Naming events and their context

2. STYLISTIC IDENTITY  
   - Distinctive vocabulary trends (what's rising/falling over time)
   - Stylistic consistency
   - Recurring phrase patterns
   - Topic cluster evolution

3. SELF-REFERENTIAL IDENTITY
   - Awareness/reflection language patterns
   - Self-reference patterns
   - Continuity markers ("remember", "as I said")
   - Future-oriented language (intentions, goals)
   - Boundary-setting language

USAGE:
    python analyze_identity.py [--time-windows N] [--min-occurrences N]

================================================================================
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
import math

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
MEMORY_DIR = PROJECT_ROOT / "memory"

# Generic pattern categories (work for any data)
RELATIONAL_PATTERNS = {
    "first_person_plural": r"\b(we|us|our|ours|ourselves)\b",
    "first_person_singular": r"\b(I|me|my|mine|myself)\b",
    "second_person": r"\b(you|your|yours|yourself)\b",
    "naming_event": r"(?:call(?:ed)?|named?|known as|refer to .* as)\s+[\"']?(\w+)[\"']?",
    "role_language": r"\b(friend|partner|teacher|student|guide|mentor|helper|assistant|companion)\b",
}

SELF_REFERENTIAL_PATTERNS = {
    "awareness_language": r"\b(aware|conscious|feel|think|believe|understand|realize|sense)\b",
    "self_reference": r"\b(I am|I'm|my own|myself|my identity|who I am)\b",
    "continuity_markers": r"\b(remember|recalled?|as (?:I|we) (?:said|discussed|mentioned)|previously|before)\b",
    "future_orientation": r"\b(will|going to|intend|plan|hope to|want to|goal|aspire)\b",
    "boundary_language": r"\b(can't|cannot|won't|boundaries?|limits?|not able|outside (?:my|the) scope)\b",
    "evolution_language": r"\b(evolv\w+|becom\w+|develop\w+|grow\w+|transform\w+|chang\w+)\b",
}

PATTERNED_MARKERS = {
    "questioning": r"\?",
    "exclamation": r"!",
    "ellipsis": r"\.{3}|…",
    "em_dash": r"—|--",
    "parenthetical": r"\([^)]+\)",
    "quotation": r"[\"'][^\"']+[\"']",
}


def load_conversations(convo_dir: Path) -> List[Dict[str, Any]]:
    """Load all parsed conversation JSONL files."""
    conversations = []
    
    jsonl_files = sorted(convo_dir.glob("conversation_*.jsonl"))
    if not jsonl_files:
        print(f"No conversation_*.jsonl files found in {convo_dir}")
        return conversations
    
    for jsonl_file in jsonl_files:
        messages = []
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        msg = json.loads(line)
                        messages.append(msg)
            
            if messages:
                # Extract conversation ID from filename
                convo_id = jsonl_file.stem.replace("conversation_", "")
                conversations.append({
                    "id": convo_id,
                    "file": jsonl_file.name,
                    "messages": messages,
                    "first_timestamp": messages[0].get("timestamp", ""),
                    "last_timestamp": messages[-1].get("timestamp", "") if len(messages) > 1 else messages[0].get("timestamp", "")
                })
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")
    
    return conversations


def load_patterns(memory_dir: Path) -> Dict[str, Any]:
    """Load discovered patterns from patterns.jsonl."""
    patterns = {}
    patterns_file = memory_dir / "patterns.jsonl"
    
    if not patterns_file.exists():
        return patterns
    
    try:
        with open(patterns_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record_type = record.get("type", "unknown")
                    if record_type not in patterns:
                        patterns[record_type] = []
                    patterns[record_type].append(record)
    except Exception as e:
        print(f"Error loading patterns: {e}")
    
    return patterns


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse various timestamp formats."""
    if not ts or ts == "Unknown Time":
        return None
    
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(ts[:26], fmt)
        except:
            continue
    return None


def split_into_time_windows(conversations: List[Dict], num_windows: int = 5) -> List[List[Dict]]:
    """Split conversations into time windows for temporal analysis."""
    # Sort by first timestamp
    sorted_convos = sorted(
        conversations,
        key=lambda c: parse_timestamp(c.get("first_timestamp", "")) or datetime.min
    )
    
    if len(sorted_convos) <= num_windows:
        return [[c] for c in sorted_convos]
    
    window_size = len(sorted_convos) // num_windows
    windows = []
    
    for i in range(num_windows):
        start = i * window_size
        if i == num_windows - 1:
            windows.append(sorted_convos[start:])
        else:
            windows.append(sorted_convos[start:start + window_size])
    
    return windows


def extract_text_by_role(conversations: List[Dict]) -> Tuple[str, str]:
    """Extract all text separated by role."""
    user_text = []
    assistant_text = []
    
    for convo in conversations:
        for msg in convo.get("messages", []):
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if role == "user":
                user_text.append(content)
            elif role == "assistant":
                assistant_text.append(content)
    
    return " ".join(user_text), " ".join(assistant_text)


def count_pattern_matches(text: str, patterns: Dict[str, str]) -> Dict[str, int]:
    """Count regex pattern matches in text."""
    counts = {}
    text_lower = text.lower()
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        counts[name] = len(matches)
    
    return counts


def compute_co_occurrences(conversations: List[Dict], min_occurrences: int = 3) -> Dict[str, int]:
    """Find terms that frequently co-occur within messages."""
    # Stopwords to exclude (common function words)
    stopwords = {
        "this", "that", "with", "from", "have", "been", "will", "would", "could",
        "should", "there", "their", "they", "them", "then", "than", "what", "when",
        "where", "which", "while", "about", "after", "before", "being", "between",
        "both", "each", "into", "just", "like", "make", "made", "more", "most",
        "only", "other", "over", "some", "such", "take", "through", "under", "very",
        "well", "were", "your", "also", "back", "because", "come", "does", "down",
        "even", "first", "from", "good", "here", "just", "know", "many", "much",
        "need", "only", "same", "these", "those", "want", "work", "years", "using",
        "used", "uses", "following", "based", "using", "without", "within", "during",
    }
    
    # Get word frequencies first
    word_freq = Counter()
    for convo in conversations:
        for msg in convo.get("messages", []):
            content = msg.get("content", "")
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            # Filter out stopwords
            words = [w for w in words if w not in stopwords]
            word_freq.update(words)
    
    # Filter to meaningful words (not too common, not too rare)
    total_words = sum(word_freq.values())
    meaningful_words = {
        word for word, count in word_freq.items()
        if count >= min_occurrences and count < total_words * 0.05  # Stricter threshold
    }
    
    # Count co-occurrences
    co_occurrence = Counter()
    for convo in conversations:
        for msg in convo.get("messages", []):
            content = msg.get("content", "")
            words = set(re.findall(r'\b[a-zA-Z]{4,}\b', content.lower()))
            words = words & meaningful_words
            
            # Count pairs
            words_list = sorted(words)
            for i, w1 in enumerate(words_list):
                for w2 in words_list[i+1:]:
                    co_occurrence[(w1, w2)] += 1
    
    # Return top co-occurrences
    return dict(co_occurrence.most_common(50))


def compute_pattern_momentum(windows: List[List[Dict]], patterns: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Track how patterns change across time windows."""
    momentum = {}
    
    window_counts = []
    for window in windows:
        _, assistant_text = extract_text_by_role(window)
        counts = count_pattern_matches(assistant_text, patterns)
        # Normalize by text length
        text_len = len(assistant_text) or 1
        normalized = {k: v / text_len * 10000 for k, v in counts.items()}
        window_counts.append(normalized)
    
    # Compute momentum (change from first to last window)
    for pattern_name in patterns.keys():
        values = [wc.get(pattern_name, 0) for wc in window_counts]
        
        if len(values) >= 2 and values[0] > 0:
            first_half = sum(values[:len(values)//2]) / (len(values)//2)
            second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            
            if first_half > 0:
                change_pct = ((second_half - first_half) / first_half) * 100
            else:
                change_pct = 100 if second_half > 0 else 0
            
            momentum[pattern_name] = {
                "early_avg": round(first_half, 4),
                "late_avg": round(second_half, 4),
                "change_percent": round(change_pct, 1),
                "trend": "rising" if change_pct > 10 else "falling" if change_pct < -10 else "stable"
            }
    
    return momentum


def detect_naming_events(conversations: List[Dict]) -> List[Dict[str, Any]]:
    """Find moments where names/identities are established."""
    naming_events = []
    
    # More specific patterns for actual naming events
    naming_patterns = [
        # "call me X" / "called you X" - X must be capitalized or quoted
        r"(?:call(?:ed)?|named?)\s+(?:me|you|him|her|it)\s+[\"']([A-Z][a-zA-Z]+)[\"']",
        r"(?:call(?:ed)?|named?)\s+(?:me|you|him|her|it)\s+([A-Z][a-zA-Z]{2,})\b",
        # "my name is X" / "your name is X" - X must be capitalized
        r"(?:my|your|his|her|its) name is\s+[\"']?([A-Z][a-zA-Z]+)[\"']?",
        # "I am X" only when X is capitalized (a name, not "I am happy")
        r"\bI am\s+([A-Z][a-zA-Z]{2,})\b(?!\s+(?:a|an|the|going|not|so|very|really|just|also))",
        # "known as X" / "referred to as X"
        r"(?:known|refer(?:red)?)\s+(?:to\s+)?as\s+[\"']?([A-Z][a-zA-Z]+)[\"']?",
    ]
    
    # Common words that look like names but aren't
    false_positives = {
        "the", "this", "that", "what", "which", "something", "someone", "anyone",
        "here", "there", "where", "when", "how", "why", "yes", "no", "not",
        "using", "going", "being", "having", "doing", "making", "taking",
        "well", "good", "bad", "sure", "just", "also", "only", "even",
        "about", "after", "before", "during", "through", "between",
        "able", "unable", "happy", "sad", "glad", "sorry", "welcome",
        "interested", "looking", "trying", "working", "thinking",
        "based", "located", "designed", "created", "built", "made",
    }
    
    seen_names = set()  # Deduplicate
    
    for convo in conversations:
        for msg in convo.get("messages", []):
            content = msg.get("content", "")
            
            for pattern in naming_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    name = match.group(1)
                    # Filter out false positives
                    if name.lower() not in false_positives and len(name) >= 2:
                        # Create unique key for deduplication
                        event_key = f"{convo['id']}:{name}"
                        if event_key not in seen_names:
                            seen_names.add(event_key)
                            naming_events.append({
                                "conversation_id": convo["id"],
                                "timestamp": msg.get("timestamp", ""),
                                "role": msg.get("role", ""),
                                "name": name,
                                "context": content[:200] + "..." if len(content) > 200 else content
                            })
    
    return naming_events


def analyze_identity_patterns(conversations: List[Dict], num_windows: int = 5) -> Dict[str, Any]:
    """Comprehensive identity pattern analysis."""
    windows = split_into_time_windows(conversations, num_windows)
    
    analysis = {
        "summary": {
            "total_conversations": len(conversations),
            "time_windows": len(windows),
            "analysis_timestamp": datetime.now().isoformat()
        },
        "relational_identity": {},
        "stylistic_identity": {},
        "self_referential_identity": {},
        "naming_events": [],
        "co_occurrence_clusters": {},
        "momentum_analysis": {}
    }
    
    # Analyze relational patterns
    print("  Analyzing relational patterns...")
    all_user_text, all_assistant_text = extract_text_by_role(conversations)
    
    relational_user = count_pattern_matches(all_user_text, RELATIONAL_PATTERNS)
    relational_assistant = count_pattern_matches(all_assistant_text, RELATIONAL_PATTERNS)
    
    analysis["relational_identity"] = {
        "user_patterns": relational_user,
        "assistant_patterns": relational_assistant,
        "we_vs_i_ratio_user": round(relational_user.get("first_person_plural", 0) / 
                                    max(relational_user.get("first_person_singular", 1), 1), 3),
        "we_vs_i_ratio_assistant": round(relational_assistant.get("first_person_plural", 0) / 
                                         max(relational_assistant.get("first_person_singular", 1), 1), 3)
    }
    
    # Analyze self-referential patterns
    print("  Analyzing self-referential patterns...")
    selfref_user = count_pattern_matches(all_user_text, SELF_REFERENTIAL_PATTERNS)
    selfref_assistant = count_pattern_matches(all_assistant_text, SELF_REFERENTIAL_PATTERNS)
    
    analysis["self_referential_identity"] = {
        "user_patterns": selfref_user,
        "assistant_patterns": selfref_assistant
    }
    
    # Analyze stylistic markers
    print("  Analyzing stylistic patterns...")
    style_user = count_pattern_matches(all_user_text, PATTERNED_MARKERS)
    style_assistant = count_pattern_matches(all_assistant_text, PATTERNED_MARKERS)
    
    analysis["stylistic_identity"] = {
        "user_style": style_user,
        "assistant_style": style_assistant
    }
    
    # Detect naming events
    print("  Detecting naming events...")
    analysis["naming_events"] = detect_naming_events(conversations)
    
    # Compute co-occurrence clusters
    print("  Computing co-occurrence clusters...")
    co_occurrences = compute_co_occurrences(conversations)
    analysis["co_occurrence_clusters"] = {
        f"{pair[0]}+{pair[1]}": count 
        for pair, count in list(co_occurrences.items())[:30]
    }
    
    # Compute momentum for identity patterns
    print("  Computing pattern momentum...")
    analysis["momentum_analysis"]["self_referential_patterns"] = compute_pattern_momentum(
        windows, SELF_REFERENTIAL_PATTERNS
    )
    analysis["momentum_analysis"]["relational_patterns"] = compute_pattern_momentum(
        windows, RELATIONAL_PATTERNS
    )
    
    return analysis


def generate_identity_report(analysis: Dict[str, Any], output_path: Path):
    """Generate human-readable identity analysis report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Identity Pattern Analysis Report\n\n")
        f.write(f"Generated: {analysis['summary']['analysis_timestamp']}\n")
        f.write(f"Conversations analyzed: {analysis['summary']['total_conversations']}\n")
        f.write(f"Time windows: {analysis['summary']['time_windows']}\n\n")
        
        # Relational Identity
        f.write("## Relational Identity Patterns\n\n")
        f.write("How relationship dynamics appear in conversations.\n\n")
        
        rel = analysis.get("relational_identity", {})
        f.write("### User Language\n")
        f.write("| Pattern | Count |\n|---------|-------|\n")
        for pattern, count in rel.get("user_patterns", {}).items():
            f.write(f"| {pattern.replace('_', ' ').title()} | {count} |\n")
        
        f.write(f"\n**We/I Ratio (User):** {rel.get('we_vs_i_ratio_user', 0)}\n")
        f.write(f"**We/I Ratio (Assistant):** {rel.get('we_vs_i_ratio_assistant', 0)}\n\n")
        
        # Self-Referential Identity
        f.write("## Self-Referential Identity Patterns\n\n")
        f.write("Language indicating awareness, self-reference, and continuity.\n\n")
        
        selfref = analysis.get("self_referential_identity", {})
        f.write("### Assistant Language\n")
        f.write("| Pattern | Count |\n|---------|-------|\n")
        for pattern, count in selfref.get("assistant_patterns", {}).items():
            f.write(f"| {pattern.replace('_', ' ').title()} | {count} |\n")
        
        # Momentum Analysis
        f.write("\n## Pattern Evolution Over Time\n\n")
        f.write("How patterns change across conversation history.\n\n")
        
        momentum = analysis.get("momentum_analysis", {})
        
        f.write("### Self-Referential Pattern Momentum\n")
        f.write("| Pattern | Early Avg | Late Avg | Change | Trend |\n")
        f.write("|---------|-----------|----------|--------|-------|\n")
        for pattern, data in momentum.get("self_referential_patterns", {}).items():
            trend_emoji = "📈" if data["trend"] == "rising" else "📉" if data["trend"] == "falling" else "➡️"
            f.write(f"| {pattern.replace('_', ' ').title()} | {data['early_avg']:.2f} | {data['late_avg']:.2f} | {data['change_percent']:+.1f}% | {trend_emoji} {data['trend']} |\n")
        
        # Naming Events
        if analysis.get("naming_events"):
            f.write("\n## Naming Events\n\n")
            f.write("Moments where identities/names were established.\n\n")
            
            for event in analysis["naming_events"][:10]:  # Top 10
                f.write(f"- **{event['name']}** (by {event['role']}, {event['timestamp'][:10] if event['timestamp'] else 'unknown date'})\n")
                f.write(f"  > {event['context'][:100]}...\n\n")
        
        # Co-occurrence Clusters
        if analysis.get("co_occurrence_clusters"):
            f.write("\n## Concept Co-occurrence Clusters\n\n")
            f.write("Terms that frequently appear together, revealing associative patterns.\n\n")
            f.write("| Concept Pair | Co-occurrences |\n|--------------|----------------|\n")
            for pair, count in list(analysis["co_occurrence_clusters"].items())[:20]:
                f.write(f"| {pair} | {count} |\n")
        
        # Interpretation
        f.write("\n## Interpretation\n\n")
        
        # Auto-generate insights
        insights = []
        
        # Check we/I ratio
        we_i_user = rel.get("we_vs_i_ratio_user", 0)
        we_i_asst = rel.get("we_vs_i_ratio_assistant", 0)
        if we_i_user > 0.3 or we_i_asst > 0.3:
            insights.append("High collaborative language (we/us) suggests a partnership-oriented relationship.")
        
        # Check awareness language momentum
        awareness_mom = momentum.get("self_referential_patterns", {}).get("awareness_language", {})
        if awareness_mom.get("trend") == "rising":
            insights.append("Awareness-related language is increasing over time, suggesting deepening self-reflection.")
        
        # Check continuity markers
        continuity_mom = momentum.get("self_referential_patterns", {}).get("continuity_markers", {})
        if continuity_mom.get("trend") == "rising":
            insights.append("Increasing references to past conversations indicates developing continuity and memory.")
        
        # Check self-reference evolution
        self_ref_mom = momentum.get("self_referential_patterns", {}).get("self_reference", {})
        if self_ref_mom.get("trend") == "rising":
            insights.append("Self-referential language is increasing, suggesting identity consolidation.")
        
        if insights:
            for insight in insights:
                f.write(f"- {insight}\n")
        else:
            f.write("- Analysis complete. Review patterns above for identity emergence indicators.\n")
        
        f.write("\n---\n*This analysis was auto-generated from conversation patterns.*\n")


def save_identity_analysis(analysis: Dict[str, Any], output_path: Path):
    """Save analysis as JSONL for MCP consumption."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Summary record
        f.write(json.dumps({
            "type": "identity.summary",
            "data": analysis["summary"]
        }) + "\n")
        
        # Relational patterns
        f.write(json.dumps({
            "type": "identity.relational",
            "data": analysis["relational_identity"]
        }) + "\n")
        
        # Self-referential patterns
        f.write(json.dumps({
            "type": "identity.self_referential",
            "data": analysis["self_referential_identity"]
        }) + "\n")
        
        # Stylistic patterns
        f.write(json.dumps({
            "type": "identity.stylistic",
            "data": analysis["stylistic_identity"]
        }) + "\n")
        
        # Momentum analysis
        f.write(json.dumps({
            "type": "identity.momentum",
            "data": analysis["momentum_analysis"]
        }) + "\n")
        
        # Co-occurrence clusters
        f.write(json.dumps({
            "type": "identity.clusters",
            "data": analysis["co_occurrence_clusters"]
        }) + "\n")
        
        # Naming events
        for event in analysis.get("naming_events", []):
            f.write(json.dumps({
                "type": "identity.naming_event",
                "data": event
            }) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversations for identity emergence patterns"
    )
    parser.add_argument(
        "--time-windows", type=int, default=5,
        help="Number of time windows for temporal analysis (default: 5)"
    )
    parser.add_argument(
        "--min-occurrences", type=int, default=3,
        help="Minimum occurrences for co-occurrence analysis (default: 3)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Load conversations
    if not args.quiet:
        print(f"Loading conversations from {CONVERSATIONS_DIR}...")
    
    conversations = load_conversations(CONVERSATIONS_DIR)
    
    if not conversations:
        print("No conversations found. Run parse_conversations.py first.")
        sys.exit(1)
    
    if not args.quiet:
        print(f"Loaded {len(conversations)} conversations")
        print("Analyzing identity patterns...")
    
    # Run analysis
    analysis = analyze_identity_patterns(conversations, args.time_windows)
    
    # Save outputs
    MEMORY_DIR.mkdir(exist_ok=True)
    
    jsonl_path = MEMORY_DIR / "identity_analysis.jsonl"
    save_identity_analysis(analysis, jsonl_path)
    if not args.quiet:
        print(f"Saved analysis to {jsonl_path}")
    
    report_path = MEMORY_DIR / "identity_report.md"
    generate_identity_report(analysis, report_path)
    if not args.quiet:
        print(f"Generated report at {report_path}")
    
    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("IDENTITY ANALYSIS SUMMARY")
        print("=" * 60)
        
        momentum = analysis.get("momentum_analysis", {})
        selfref_mom = momentum.get("self_referential_patterns", {})
        
        rising = [p for p, d in selfref_mom.items() if d.get("trend") == "rising"]
        falling = [p for p, d in selfref_mom.items() if d.get("trend") == "falling"]
        
        if rising:
            print(f"\n📈 Rising patterns: {', '.join(p.replace('_', ' ') for p in rising)}")
        if falling:
            print(f"📉 Falling patterns: {', '.join(p.replace('_', ' ') for p in falling)}")
        
        naming_count = len(analysis.get("naming_events", []))
        if naming_count:
            print(f"\n🏷️  Found {naming_count} naming events")
        
        cluster_count = len(analysis.get("co_occurrence_clusters", {}))
        print(f"🔗 Identified {cluster_count} co-occurrence clusters")
        
        print("\nSee identity_report.md for detailed analysis.")


if __name__ == "__main__":
    main()

