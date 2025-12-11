#!/usr/bin/env python3
"""
Pattern Analyzer for Conversation Data
=======================================

Analyzes your ChatGPT conversations to discover patterns unique to YOUR data.
Outputs identity and pattern files that other scripts can use automatically.

Data Sources:
  - conversations/*.jsonl: Processed conversation files (preferred)
  - conversations/conversations.json: Raw ChatGPT export (fallback)
  - files/: RAG storage files (optional)

Outputs:
  - memory/identity.jsonl: Core identity patterns
  - memory/patterns.jsonl: Keywords, topics, tones, entities for other scripts

Usage:
  python analyze_patterns.py                # Analyze all conversations
  python analyze_patterns.py --sample 100   # Sample 100 conversations
  python analyze_patterns.py --min-freq 10  # Higher frequency threshold
  python analyze_patterns.py --no-files     # Skip files/ directory
"""

import json
import os
import sys
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from datetime import datetime
import random

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
FILES_DIR = get_user_dir(PROJECT_ROOT / "files", USER_ID)
MEMORY_DIR = get_user_dir(PROJECT_ROOT / "memory", USER_ID)

# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'you',
    'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'would', 'could', 'might', 'must', 'shall', 'if', 'your', 'my', 'our',
    'their', 'its', 'his', 'her', 'me', 'him', 'us', 'them', 'as', 'also',
    've', 'll', 're', 'd', 'm', 'like', 'get', 'got', 'one', 'two', 'first',
    'make', 'use', 'using', 'used', 'way', 'know', 'think', 'see', 'want',
    'need', 'look', 'time', 'new', 'good', 'well', 'back', 'even', 'because',
    'any', 'give', 'day', 'take', 'come', 'over', 'still', 'let', 'say',
    'something', 'thing', 'things', 'try', 'much', 'going', 'may', 'yes',
    'really', 'actually', 'basically', 'kind', 'sort', 'right', 'sure',
    'okay', 'maybe', 'probably', 'definitely', 'always', 'never', 'sometimes'
}


def load_conversations(conversations_dir: Path, sample_size: Optional[int] = None) -> List[Dict]:
    """Load conversations from JSONL files or raw conversations.json."""
    conversations = []
    
    # Try JSONL files first
    jsonl_files = list(conversations_dir.glob("conversation_*.jsonl"))
    if not jsonl_files:
        jsonl_files = list(conversations_dir.glob("*.jsonl"))
    
    if jsonl_files:
        if sample_size and len(jsonl_files) > sample_size:
            jsonl_files = random.sample(jsonl_files, sample_size)
        
        print(f"Loading {len(jsonl_files)} conversation files...")
        
        for file_path in jsonl_files:
            messages = []
            try:
                for line in file_path.read_text(encoding='utf-8').split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                
                if messages:
                    conversations.append({
                        'id': file_path.stem,
                        'file': str(file_path),
                        'messages': messages
                    })
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
        
        if conversations:
            print(f"Loaded {len(conversations)} conversations")
            return conversations
    
    # Fall back to raw conversations.json
    conversations_json = conversations_dir / "conversations.json"
    if conversations_json.exists():
        print(f"Loading from {conversations_json}...")
        try:
            with open(conversations_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_convos = data if isinstance(data, list) else [data]
            
            for conv in raw_convos:
                messages = []
                if 'mapping' in conv:
                    for node_id, node in conv.get('mapping', {}).items():
                        msg = node.get('message')
                        if msg and msg.get('content') and msg.get('author'):
                            role = msg['author'].get('role', 'unknown')
                            content_parts = msg['content'].get('parts', [])
                            content = ' '.join(str(p) for p in content_parts if p)
                            if content and role in ('user', 'assistant'):
                                messages.append({
                                    'role': role,
                                    'content': content,
                                    'timestamp': msg.get('create_time', '')
                                })
                elif 'messages' in conv:
                    messages = conv['messages']
                
                if messages:
                    conv_id = conv.get('id', conv.get('conversation_id', str(len(conversations))))
                    conversations.append({
                        'id': conv_id,
                        'title': conv.get('title', ''),
                        'messages': messages
                    })
            
            if sample_size and len(conversations) > sample_size:
                conversations = random.sample(conversations, sample_size)
            
            print(f"Loaded {len(conversations)} conversations")
            return conversations
        except Exception as e:
            print(f"Warning: Error loading conversations.json: {e}")
    
    print("No conversations found!")
    return conversations


def load_files(files_dir: Path) -> List[Dict]:
    """Load text files from files/ directory."""
    files_data = []
    if not files_dir.exists():
        return []
    
    valid_extensions = {'.txt', '.md', '.json', '.jsonl', '.yaml', '.yml'}
    
    for root, dirs, files in os.walk(files_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in files:
            if filename.startswith('.') or 'Zone.Identifier' in filename:
                continue
            
            ext = Path(filename).suffix.lower()
            if ext not in valid_extensions:
                continue
            
            file_path = Path(root) / filename
            try:
                content = file_path.read_text(encoding='utf-8')
                files_data.append({
                    'filepath': str(file_path.relative_to(files_dir)),
                    'content': content
                })
            except Exception:
                continue
    
    if files_data:
        print(f"Loaded {len(files_data)} files from files/")
    return files_data


def extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z\'-]*[a-zA-Z]\b|\b[a-zA-Z]\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def extract_capitalized_words(text: str) -> List[str]:
    """Extract capitalized words (potential names)."""
    text = re.sub(r'```[\s\S]*?```', '', text)
    words = []
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > 1:
            for word in sentence_words[1:]:
                match = re.match(r'^[A-Z][a-z]{2,}$', word)
                if match:
                    words.append(match.group())
    return words


def analyze_conversations(conversations: List[Dict], files_data: List[Dict]) -> Dict:
    """Analyze all data to discover patterns."""
    analysis = {
        'word_freq': Counter(),
        'user_word_freq': Counter(),
        'assistant_word_freq': Counter(),
        'bigram_freq': Counter(),
        'trigram_freq': Counter(),
        'capitalized_words': Counter(),
        'emojis': Counter(),
        'phrase_patterns': Counter(),
        'message_count': 0,
        'user_message_count': 0,
        'assistant_message_count': 0,
        'conversation_count': len(conversations),
    }
    
    print("Analyzing patterns...")
    
    # Analyze conversations
    for conv in conversations:
        for msg in conv.get('messages', []):
            content = msg.get('content', '')
            if not content or not isinstance(content, str):
                continue
            
            analysis['message_count'] += 1
            role = msg.get('role', '')
            
            if role == 'user':
                analysis['user_message_count'] += 1
            elif role == 'assistant':
                analysis['assistant_message_count'] += 1
            
            words = extract_words(content)
            analysis['word_freq'].update(words)
            
            if role == 'user':
                analysis['user_word_freq'].update(words)
            elif role == 'assistant':
                analysis['assistant_word_freq'].update(words)
            
            if len(words) >= 2:
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                analysis['bigram_freq'].update(bigrams)
            if len(words) >= 3:
                trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
                analysis['trigram_freq'].update(trigrams)
            
            if role == 'assistant':
                caps = extract_capitalized_words(content)
                analysis['capitalized_words'].update(caps)
            
            emojis = re.findall(r'[\U0001F300-\U0001F9FF]', content)
            analysis['emojis'].update(emojis)
            
            # Phrase patterns
            if role == 'user':
                for match in re.findall(r'\bi (\w+)\b', content.lower())[:3]:
                    if match not in STOPWORDS:
                        analysis['phrase_patterns'].update([f"i {match}"])
            if role == 'assistant':
                for match in re.findall(r'\byou (\w+)\b', content.lower())[:3]:
                    if match not in STOPWORDS:
                        analysis['phrase_patterns'].update([f"you {match}"])
    
    # Analyze files
    for file_info in files_data:
        content = file_info.get('content', '')
        if not content:
            continue
        
        words = extract_words(content)
        analysis['word_freq'].update(words)
        
        if len(words) >= 2:
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            analysis['bigram_freq'].update(bigrams)
        
        caps = extract_capitalized_words(content)
        analysis['capitalized_words'].update(caps)
        
        emojis = re.findall(r'[\U0001F300-\U0001F9FF]', content)
        analysis['emojis'].update(emojis)
    
    return analysis


def find_distinctive_terms(analysis: Dict, min_freq: int = 5) -> List[Dict]:
    """Find statistically distinctive terms."""
    word_freq = analysis['word_freq']
    total_words = sum(word_freq.values())
    
    distinctive = []
    for word, count in word_freq.items():
        if count < min_freq:
            continue
        
        freq_ratio = count / total_words
        if freq_ratio > 0.01:  # Skip very common words
            continue
        
        user_count = analysis['user_word_freq'].get(word, 0)
        asst_count = analysis['assistant_word_freq'].get(word, 0)
        
        source = "both"
        if user_count > asst_count * 2:
            source = "user"
        elif asst_count > user_count * 2:
            source = "assistant"
        
        distinctive.append({
            'word': word,
            'count': count,
            'frequency': freq_ratio,
            'source': source
        })
    
    distinctive.sort(key=lambda x: x['count'], reverse=True)
    return distinctive[:150]


def find_recurring_phrases(analysis: Dict, min_freq: int = 3) -> List[Dict]:
    """Find recurring multi-word phrases."""
    phrases = []
    
    for bigram, count in analysis['bigram_freq'].most_common(100):
        if count >= min_freq:
            phrases.append({'phrase': bigram, 'count': count, 'type': 'bigram'})
    
    for trigram, count in analysis['trigram_freq'].most_common(50):
        if count >= min_freq:
            phrases.append({'phrase': trigram, 'count': count, 'type': 'trigram'})
    
    return phrases


def find_entities(analysis: Dict, min_freq: int = 3) -> List[Dict]:
    """Find potential names/entities."""
    entities = []
    for word, count in analysis['capitalized_words'].most_common(50):
        if count >= min_freq:
            entities.append({'name': word, 'count': count})
    return entities


def detect_topic_clusters(analysis: Dict, distinctive_terms: List[Dict]) -> Dict[str, List[str]]:
    """Auto-detect topic clusters from word co-occurrence."""
    # Group words that frequently appear together
    bigram_words = defaultdict(set)
    for bigram, count in analysis['bigram_freq'].most_common(200):
        if count >= 3:
            w1, w2 = bigram.split()
            bigram_words[w1].add(w2)
            bigram_words[w2].add(w1)
    
    # Find clusters based on connectivity
    top_words = [t['word'] for t in distinctive_terms[:50]]
    clusters = {}
    
    # Technical cluster
    tech_seeds = ['code', 'api', 'function', 'error', 'bug', 'server', 'database', 'python', 'javascript']
    tech_cluster = [w for w in top_words if w in tech_seeds or any(s in bigram_words.get(w, set()) for s in tech_seeds)]
    if tech_cluster:
        clusters['technical'] = tech_cluster[:15]
    
    # Emotional cluster
    emot_seeds = ['feel', 'feeling', 'emotion', 'pain', 'fear', 'anxiety', 'happy', 'sad', 'love', 'hate']
    emot_cluster = [w for w in top_words if w in emot_seeds or any(s in bigram_words.get(w, set()) for s in emot_seeds)]
    if emot_cluster:
        clusters['emotional'] = emot_cluster[:15]
    
    # Creative cluster
    creative_seeds = ['story', 'character', 'write', 'writing', 'creative', 'fiction', 'narrative', 'poem']
    creative_cluster = [w for w in top_words if w in creative_seeds or any(s in bigram_words.get(w, set()) for s in creative_seeds)]
    if creative_cluster:
        clusters['creative'] = creative_cluster[:15]
    
    # Meta/philosophical cluster
    meta_seeds = ['consciousness', 'awareness', 'meaning', 'existence', 'reality', 'truth', 'identity']
    meta_cluster = [w for w in top_words if w in meta_seeds or any(s in bigram_words.get(w, set()) for s in meta_seeds)]
    if meta_cluster:
        clusters['philosophical'] = meta_cluster[:15]
    
    return clusters


def detect_tone_indicators(analysis: Dict) -> Dict[str, List[str]]:
    """Detect tone indicators from phrase patterns."""
    tones = {
        'personal': [],
        'analytical': [],
        'exploratory': [],
        'distressed': [],
        'playful': []
    }
    
    # Map phrase patterns to tones
    personal_markers = ['i feel', 'i think', 'i believe', 'i want', 'i need', 'my experience']
    analytical_markers = ['analyze', 'evaluate', 'compare', 'understand', 'explain']
    exploratory_markers = ['what if', 'explore', 'imagine', 'consider', 'wonder']
    distressed_markers = ['overwhelmed', 'struggling', 'difficult', 'pain', 'hard']
    playful_markers = ['haha', 'lol', 'fun', 'joke', 'play']
    
    word_freq = analysis['word_freq']
    
    for word, count in word_freq.most_common(500):
        if any(m in word for m in personal_markers):
            tones['personal'].append(word)
        if any(m in word for m in analytical_markers):
            tones['analytical'].append(word)
        if any(m in word for m in exploratory_markers):
            tones['exploratory'].append(word)
        if any(m in word for m in distressed_markers):
            tones['distressed'].append(word)
        if any(m in word for m in playful_markers):
            tones['playful'].append(word)
    
    # Also check phrase patterns
    for phrase, count in analysis['phrase_patterns'].most_common(100):
        if any(m in phrase for m in personal_markers):
            tones['personal'].append(phrase)
    
    return {k: v[:10] for k, v in tones.items() if v}


def generate_identity_jsonl(analysis: Dict, distinctive_terms: List[Dict], 
                            phrases: List[Dict], entities: List[Dict]) -> str:
    """Generate identity.jsonl."""
    records = []
    timestamp = datetime.now().isoformat()
    
    # Core stats
    records.append({
        "id": "identity-001",
        "type": "identity.core",
        "tags": ["origin", "statistics"],
        "content": f"This identity was built from {analysis['conversation_count']} conversations containing {analysis['message_count']} messages.",
        "stats": {
            "conversations": analysis['conversation_count'],
            "messages": analysis['message_count'],
            "user_messages": analysis['user_message_count'],
            "assistant_messages": analysis['assistant_message_count']
        },
        "createdAt": timestamp
    })
    
    # Vocabulary
    top_terms = [t['word'] for t in distinctive_terms[:20]]
    if top_terms:
        records.append({
            "id": "identity-002",
            "type": "identity.vocabulary",
            "tags": ["distinctive", "vocabulary"],
            "content": f"Key vocabulary: {', '.join(top_terms[:10])}",
            "terms": top_terms,
            "createdAt": timestamp
        })
    
    # Entities
    if entities:
        entity_names = [e['name'] for e in entities[:10]]
        records.append({
            "id": "identity-003",
            "type": "identity.entities",
            "tags": ["names", "entities"],
            "content": f"Recurring names/entities: {', '.join(entity_names)}",
            "entities": entity_names,
            "createdAt": timestamp
        })
    
    # User voice
    user_patterns = [p for p, c in analysis['phrase_patterns'].most_common(20) if p.startswith('i ')]
    if user_patterns:
        records.append({
            "id": "identity-004",
            "type": "identity.user_voice",
            "tags": ["user", "patterns"],
            "content": f"User frequently says: {', '.join(user_patterns[:5])}",
            "patterns": user_patterns,
            "createdAt": timestamp
        })
    
    return '\n'.join(json.dumps(r) for r in records) + '\n'


def generate_patterns_jsonl(analysis: Dict, distinctive_terms: List[Dict],
                            phrases: List[Dict], entities: List[Dict],
                            topic_clusters: Dict, tone_indicators: Dict) -> str:
    """Generate patterns.jsonl with everything other scripts need."""
    records = []
    timestamp = datetime.now().isoformat()
    
    # Keywords for tagging (used by parse_memories.py and build_interaction_map.py)
    keywords = [t['word'] for t in distinctive_terms[:50]]
    records.append({
        "id": "pattern-keywords",
        "type": "pattern.keywords",
        "tags": ["keywords", "tagging"],
        "content": f"Distinctive keywords discovered: {', '.join(keywords[:20])}",
        "keywords": keywords,
        "createdAt": timestamp
    })
    
    # Topic clusters (used by build_interaction_map.py)
    if topic_clusters:
        records.append({
            "id": "pattern-topics",
            "type": "pattern.topics",
            "tags": ["topics", "categories"],
            "content": f"Topic categories detected: {', '.join(topic_clusters.keys())}",
            "topics": topic_clusters,
            "createdAt": timestamp
        })
    
    # Tone indicators (used by build_interaction_map.py)
    if tone_indicators:
        records.append({
            "id": "pattern-tones",
            "type": "pattern.tones",
            "tags": ["tone", "emotional"],
            "content": f"Tone indicators: {', '.join(tone_indicators.keys())}",
            "tones": tone_indicators,
            "createdAt": timestamp
        })
    
    # Entity names (used by build_interaction_map.py for pattern detection)
    if entities:
        entity_names = [e['name'] for e in entities]
        records.append({
            "id": "pattern-entities",
            "type": "pattern.entities",
            "tags": ["names", "entities"],
            "content": f"Detected names/entities: {', '.join(entity_names[:10])}",
            "entities": entity_names,
            "createdAt": timestamp
        })
    
    # Recurring phrases
    if phrases:
        top_phrases = [p['phrase'] for p in phrases[:30]]
        records.append({
            "id": "pattern-phrases",
            "type": "pattern.phrases",
            "tags": ["phrases", "recurring"],
            "content": f"Recurring phrases: {', '.join(top_phrases[:10])}",
            "phrases": top_phrases,
            "createdAt": timestamp
        })
    
    # Emojis (used for symbolic detection)
    if analysis['emojis']:
        top_emojis = [e for e, c in analysis['emojis'].most_common(20)]
        records.append({
            "id": "pattern-emojis",
            "type": "pattern.emojis",
            "tags": ["emoji", "symbolic"],
            "content": f"Common emojis: {' '.join(top_emojis[:10])}",
            "emojis": top_emojis,
            "createdAt": timestamp
        })
    
    # Hub words (central concepts)
    bigram_words = defaultdict(list)
    for bigram, count in analysis['bigram_freq'].most_common(100):
        if count >= 3:
            w1, w2 = bigram.split()
            bigram_words[w1].append(w2)
            bigram_words[w2].append(w1)
    
    hub_words = [(w, len(assoc)) for w, assoc in bigram_words.items() if len(assoc) >= 5]
    hub_words.sort(key=lambda x: x[1], reverse=True)
    
    if hub_words:
        records.append({
            "id": "pattern-hubs",
            "type": "pattern.hub_concepts",
            "tags": ["concepts", "semantic"],
            "content": f"Central concepts: {', '.join(w for w, c in hub_words[:10])}",
            "hubs": [{"word": w, "connections": c} for w, c in hub_words[:15]],
            "createdAt": timestamp
        })
    
    # Meta record
    records.append({
        "id": "pattern-meta",
        "type": "pattern.meta",
        "tags": ["hydration"],
        "content": "Patterns discovered through statistical analysis. Other scripts (build_interaction_map.py, parse_memories.py) read this file for configuration.",
        "createdAt": timestamp
    })
    
    return '\n'.join(json.dumps(r) for r in records) + '\n'


def print_report(analysis: Dict, distinctive_terms: List[Dict],
                 phrases: List[Dict], entities: List[Dict]):
    """Print analysis report."""
    print("\n" + "="*60)
    print("PATTERN ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 STATISTICS")
    print(f"   Conversations: {analysis['conversation_count']}")
    print(f"   Messages: {analysis['message_count']} (user: {analysis['user_message_count']}, assistant: {analysis['assistant_message_count']})")
    print(f"   Unique words: {len(analysis['word_freq'])}")
    
    print(f"\n📝 TOP DISTINCTIVE TERMS")
    for i, term in enumerate(distinctive_terms[:15], 1):
        icon = "👤" if term['source'] == 'user' else "🤖" if term['source'] == 'assistant' else "↔️"
        print(f"   {i:2d}. {term['word']}: {term['count']} {icon}")
    
    print(f"\n🔗 RECURRING PHRASES")
    for phrase in phrases[:10]:
        print(f"   - \"{phrase['phrase']}\": {phrase['count']}x")
    
    print(f"\n👤 POTENTIAL NAMES/ENTITIES")
    if entities:
        for entity in entities[:8]:
            print(f"   - {entity['name']}: {entity['count']}x")
    else:
        print("   (None detected)")
    
    if analysis['emojis']:
        print(f"\n😀 TOP EMOJIS")
        emoji_str = " ".join([f"{e}({c})" for e, c in analysis['emojis'].most_common(8)])
        print(f"   {emoji_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Discover patterns in your conversation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--conversations-dir', type=str, default=None)
    parser.add_argument('--files-dir', type=str, default=None)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--min-freq', type=int, default=5)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--no-files', action='store_true')
    
    args = parser.parse_args()
    
    conv_dir = Path(args.conversations_dir) if args.conversations_dir else CONVERSATIONS_DIR
    files_dir = Path(args.files_dir) if args.files_dir else FILES_DIR
    
    # Load data
    conversations = []
    if conv_dir.exists():
        conversations = load_conversations(conv_dir, sample_size=args.sample)
    
    files_data = []
    if not args.no_files and files_dir.exists():
        files_data = load_files(files_dir)
    
    if not conversations and not files_data:
        print("Error: No data to analyze.")
        sys.exit(1)
    
    # Analyze
    analysis = analyze_conversations(conversations, files_data)
    
    # Find patterns
    distinctive_terms = find_distinctive_terms(analysis, min_freq=args.min_freq)
    phrases = find_recurring_phrases(analysis, min_freq=max(3, args.min_freq // 2))
    entities = find_entities(analysis, min_freq=max(3, args.min_freq // 2))
    topic_clusters = detect_topic_clusters(analysis, distinctive_terms)
    tone_indicators = detect_tone_indicators(analysis)
    
    # Report
    if not args.quiet:
        print_report(analysis, distinctive_terms, phrases, entities)
    
    # Output
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # identity.jsonl
    identity_content = generate_identity_jsonl(analysis, distinctive_terms, phrases, entities)
    identity_path = MEMORY_DIR / "identity.jsonl"
    with open(identity_path, 'w', encoding='utf-8') as f:
        f.write(identity_content)
    print(f"\n✅ Saved: {identity_path}")
    
    # patterns.jsonl (used by other scripts)
    patterns_content = generate_patterns_jsonl(analysis, distinctive_terms, phrases, entities, topic_clusters, tone_indicators)
    patterns_path = MEMORY_DIR / "patterns.jsonl"
    with open(patterns_path, 'w', encoding='utf-8') as f:
        f.write(patterns_content)
    print(f"✅ Saved: {patterns_path}")
    
    print("\n" + "-"*60)
    print("DONE! Other scripts will automatically use patterns.jsonl")
    print("-"*60)


if __name__ == '__main__':
    main()
