#!/usr/bin/env python3
"""
================================================================================
IDENTITY EMBEDDING MODEL TRAINER
================================================================================

PURPOSE:
    Trains a behavioral identity verification model from conversation data.
    Creates an "identity fingerprint" that captures how a person communicates,
    enabling verification of whether new messages match their identity.

WHAT IT LEARNS:
    1. Semantic patterns - What topics/concepts they discuss
    2. Stylistic markers - Punctuation, capitalization, message structure
    3. Vocabulary fingerprint - Distinctive word usage
    4. Relational dynamics - How they interact with AI

SYSTEM REQUIREMENTS:
    - GPU: NVIDIA GPU with 4GB+ VRAM (recommended) or CPU (slower)
    - RAM: 8GB minimum, 16GB recommended
    - Disk: ~500MB for model files
    - Python: 3.9+

DEPENDENCIES:
    pip install torch sentence-transformers numpy scikit-learn tqdm

OUTPUT:
    models/identity/
    ├── config.json           # Model configuration
    ├── identity_centroid.npy # Identity vector centroid
    ├── stylistic_profile.json # Stylistic feature statistics
    ├── vocabulary_profile.json # Word frequency fingerprint
    └── model/                # Fine-tuned sentence transformer (optional)

USAGE:
    python train_identity_model.py [--fine-tune] [--epochs N] [--device cuda/cpu]

================================================================================
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check dependencies
try:
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from tqdm import tqdm
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
MEMORY_DIR = PROJECT_ROOT / "memory"
MODELS_DIR = PROJECT_ROOT / "models" / "identity"

# Model configuration
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 110M params, good balance
ALTERNATIVE_MODELS = {
    "tiny": "sentence-transformers/all-MiniLM-L6-v2",      # 22M params, fastest
    "small": "sentence-transformers/all-MiniLM-L12-v2",    # 33M params
    "base": "sentence-transformers/all-mpnet-base-v2",     # 110M params (default)
    "large": "sentence-transformers/all-mpnet-base-v2",    # Same as base for now
}


def check_dependencies():
    """Verify all required dependencies are installed."""
    if not HAS_DEPS:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"\nError: {MISSING_DEP}")
        print("\nPlease install required packages:")
        print("  pip install torch sentence-transformers numpy scikit-learn tqdm")
        print("\nFor GPU support (recommended):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)


def load_conversations() -> List[Dict[str, Any]]:
    """Load all parsed conversation JSONL files."""
    conversations = []
    
    jsonl_files = sorted(CONVERSATIONS_DIR.glob("conversation_*.jsonl"))
    if not jsonl_files:
        print(f"No conversation files found in {CONVERSATIONS_DIR}")
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
                convo_id = jsonl_file.stem.replace("conversation_", "")
                conversations.append({
                    "id": convo_id,
                    "messages": messages
                })
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")
    
    return conversations


def load_memory_files() -> Dict[str, List[Dict]]:
    """Load all processed memory files for richer identity training."""
    memory_data = {
        "patterns": [],
        "identity": [],
        "identity_analysis": [],
        "user_context": [],
    }
    
    # Load patterns.jsonl - discovered keywords, topics, entities
    patterns_path = MEMORY_DIR / "patterns.jsonl"
    if patterns_path.exists():
        try:
            with open(patterns_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memory_data["patterns"].append(json.loads(line))
            print(f"  ✓ Loaded patterns.jsonl ({len(memory_data['patterns'])} records)")
        except Exception as e:
            print(f"  ⚠ Error loading patterns.jsonl: {e}")
    
    # Load identity.jsonl - core identity patterns
    identity_path = MEMORY_DIR / "identity.jsonl"
    if identity_path.exists():
        try:
            with open(identity_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memory_data["identity"].append(json.loads(line))
            print(f"  ✓ Loaded identity.jsonl ({len(memory_data['identity'])} records)")
        except Exception as e:
            print(f"  ⚠ Error loading identity.jsonl: {e}")
    
    # Load identity_analysis.jsonl - relational patterns, naming events
    analysis_path = MEMORY_DIR / "identity_analysis.jsonl"
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memory_data["identity_analysis"].append(json.loads(line))
            print(f"  ✓ Loaded identity_analysis.jsonl ({len(memory_data['identity_analysis'])} records)")
        except Exception as e:
            print(f"  ⚠ Error loading identity_analysis.jsonl: {e}")
    
    # Load user.context.jsonl - ChatGPT memories
    context_path = MEMORY_DIR / "user.context.jsonl"
    if context_path.exists():
        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        memory_data["user_context"].append(json.loads(line))
            print(f"  ✓ Loaded user.context.jsonl ({len(memory_data['user_context'])} records)")
        except Exception as e:
            print(f"  ⚠ Error loading user.context.jsonl: {e}")
    
    return memory_data


def extract_identity_signals(memory_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Extract identity-enhancing signals from memory files."""
    signals = {
        "distinctive_terms": set(),
        "key_topics": set(),
        "key_entities": set(),
        "identity_phrases": [],
        "relational_markers": {},
        "naming_events": [],
        "tone_patterns": {},
    }
    
    # From patterns.jsonl
    for record in memory_data.get("patterns", []):
        record_type = record.get("type", "")
        data = record.get("data", record)
        
        if "keywords" in record_type or "keywords" in data:
            keywords = data.get("keywords", data.get("data", {}).get("keywords", []))
            if isinstance(keywords, list):
                signals["distinctive_terms"].update(keywords)
        
        if "topics" in record_type or "topics" in data:
            topics = data.get("topics", data.get("data", {}).get("topics", []))
            if isinstance(topics, list):
                signals["key_topics"].update(topics)
        
        if "entities" in record_type or "entities" in data:
            entities = data.get("entities", data.get("data", {}).get("entities", []))
            if isinstance(entities, list):
                signals["key_entities"].update(entities)
        
        if "tones" in record_type or "tones" in data:
            tones = data.get("tones", data.get("data", {}).get("tones", {}))
            if isinstance(tones, dict):
                signals["tone_patterns"].update(tones)
    
    # From identity_analysis.jsonl
    for record in memory_data.get("identity_analysis", []):
        record_type = record.get("type", "")
        data = record.get("data", record)
        
        if "relational" in record_type:
            signals["relational_markers"] = data
        
        if "naming_event" in record_type:
            signals["naming_events"].append(data)
    
    # From identity.jsonl - extract key phrases
    for record in memory_data.get("identity", []):
        content = record.get("content", "")
        if content and len(content) > 20:
            signals["identity_phrases"].append(content)
    
    # From user.context.jsonl - extract as additional identity phrases
    for record in memory_data.get("user_context", []):
        content = record.get("content", record.get("text", ""))
        if content and len(content) > 20:
            signals["identity_phrases"].append(content)
    
    # Convert sets to lists for JSON serialization
    signals["distinctive_terms"] = list(signals["distinctive_terms"])
    signals["key_topics"] = list(signals["key_topics"])
    signals["key_entities"] = list(signals["key_entities"])
    
    return signals


def extract_user_messages(conversations: List[Dict]) -> List[Dict[str, Any]]:
    """Extract all user messages with metadata."""
    user_messages = []
    
    for convo in conversations:
        for msg in convo.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content and len(content.strip()) > 10:  # Skip very short messages
                    user_messages.append({
                        "content": content,
                        "timestamp": msg.get("timestamp", ""),
                        "conversation_id": convo["id"]
                    })
    
    return user_messages


def compute_stylistic_features(text: str) -> Dict[str, float]:
    """Extract stylistic features from text."""
    if not text:
        return {}
    
    # Basic counts
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.findall(r'[.!?]+', text)) or 1
    
    # Punctuation ratios
    features = {
        # Length metrics
        "avg_word_length": sum(len(w) for w in text.split()) / max(word_count, 1),
        "avg_sentence_length": word_count / sentence_count,
        "char_count": char_count,
        "word_count": word_count,
        
        # Punctuation density (per 100 chars)
        "question_marks": text.count("?") / char_count * 100,
        "exclamation_marks": text.count("!") / char_count * 100,
        "commas": text.count(",") / char_count * 100,
        "periods": text.count(".") / char_count * 100,
        "ellipsis": len(re.findall(r'\.{3}|…', text)) / char_count * 100,
        "em_dashes": len(re.findall(r'—|--', text)) / char_count * 100,
        "parentheses": len(re.findall(r'\([^)]*\)', text)) / char_count * 100,
        "quotes": len(re.findall(r'["\']', text)) / char_count * 100,
        
        # Capitalization
        "caps_ratio": sum(1 for c in text if c.isupper()) / max(char_count, 1),
        "all_caps_words": len(re.findall(r'\b[A-Z]{2,}\b', text)) / max(word_count, 1),
        
        # Structure
        "newline_ratio": text.count("\n") / max(char_count, 1),
        "code_blocks": len(re.findall(r'```', text)),
        "bullet_points": len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE)),
        "numbered_lists": len(re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE)),
        
        # Linguistic markers
        "first_person_singular": len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / max(word_count, 1),
        "first_person_plural": len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.I)) / max(word_count, 1),
        "hedging_words": len(re.findall(r'\b(maybe|perhaps|possibly|might|could|seems?|think|believe|feel)\b', text, re.I)) / max(word_count, 1),
        "certainty_words": len(re.findall(r'\b(definitely|certainly|absolutely|clearly|obviously|always|never)\b', text, re.I)) / max(word_count, 1),
    }
    
    return features


def compute_stylistic_profile(messages: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate stylistic profile from all messages."""
    all_features = []
    
    for msg in messages:
        features = compute_stylistic_features(msg["content"])
        all_features.append(features)
    
    if not all_features:
        return {}
    
    # Compute mean and std for each feature
    profile = {}
    feature_names = all_features[0].keys()
    
    for name in feature_names:
        values = [f[name] for f in all_features if name in f]
        if values:
            profile[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
    
    return profile


def compute_vocabulary_fingerprint(
    messages: List[Dict], 
    identity_signals: Optional[Dict[str, Any]] = None,
    top_n: int = 500
) -> Dict[str, Any]:
    """Create vocabulary fingerprint using TF-IDF and word frequencies.
    
    Enhanced with identity signals from processed memory files.
    """
    texts = [msg["content"] for msg in messages]
    
    # Word frequencies
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    
    # Normalize frequencies
    word_probs = {
        word: count / total_words 
        for word, count in word_freq.most_common(top_n)
    }
    
    # Boost weights for words that match discovered identity signals
    identity_boost = {}
    if identity_signals:
        distinctive = set(w.lower() for w in identity_signals.get("distinctive_terms", []))
        topics = set(w.lower() for w in identity_signals.get("key_topics", []))
        entities = set(w.lower() for w in identity_signals.get("key_entities", []))
        
        for word in word_probs:
            boost = 1.0
            if word in distinctive:
                boost *= 2.0  # Double weight for distinctive terms
            if word in topics:
                boost *= 1.5  # 1.5x for topics
            if word in entities:
                boost *= 1.5  # 1.5x for entities
            if boost > 1.0:
                identity_boost[word] = boost
    
    # Distinctive phrases (bigrams)
    bigrams = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
    
    bigram_freq = Counter(bigrams)
    top_bigrams = dict(bigram_freq.most_common(100))
    
    # TF-IDF for distinctive terms
    try:
        vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=2,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get most distinctive terms
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-50:][::-1]
        distinctive_terms = {
            feature_names[i]: float(mean_tfidf[i])
            for i in top_indices
        }
    except Exception:
        distinctive_terms = {}
    
    return {
        "word_frequencies": word_probs,
        "bigrams": top_bigrams,
        "distinctive_terms": distinctive_terms,
        "identity_boosted_terms": identity_boost,  # Terms boosted by memory analysis
        "vocabulary_size": len(word_freq),
        "total_words": total_words
    }


def compute_semantic_embeddings(
    messages: List[Dict],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """Compute semantic embeddings for all messages."""
    texts = [msg["content"] for msg in messages]
    
    print(f"  Computing embeddings for {len(texts)} messages...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def compute_identity_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute the identity centroid (mean embedding)."""
    return np.mean(embeddings, axis=0)


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse timestamp string to datetime."""
    if not ts or ts == "Unknown":
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


def compute_temporal_analysis(
    messages: List[Dict],
    embeddings: np.ndarray,
    centroid: np.ndarray
) -> Dict[str, Any]:
    """Compute temporal evolution of identity metrics."""
    if not messages or len(messages) < 2:
        return {}
    
    # Parse timestamps and sort
    message_data = []
    for i, msg in enumerate(messages):
        ts = parse_timestamp(msg.get("timestamp", ""))
        if ts:
            similarity = float(np.dot(embeddings[i], centroid) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(centroid) + 1e-8
            ))
            message_data.append({
                "timestamp": ts,
                "similarity": similarity,
                "distance": float(np.linalg.norm(embeddings[i] - centroid)),
                "content": msg.get("content", "")
            })
    
    if len(message_data) < 2:
        return {}
    
    message_data.sort(key=lambda x: x["timestamp"])
    
    # Time windows (split into 5 equal periods)
    total_span = (message_data[-1]["timestamp"] - message_data[0]["timestamp"]).total_seconds()
    if total_span < 3600:  # Less than 1 hour, use message count instead
        window_size = max(1, len(message_data) // 5)
        windows = []
        for i in range(0, len(message_data), window_size):
            windows.append(message_data[i:i+window_size])
    else:
        window_duration = total_span / 5
        windows = [[]]
        current_window_start = message_data[0]["timestamp"]
        
        for msg in message_data:
            if (msg["timestamp"] - current_window_start).total_seconds() > window_duration:
                windows.append([])
                current_window_start = msg["timestamp"]
            windows[-1].append(msg)
    
    # Compute metrics per window
    window_metrics = []
    for window in windows:
        if not window:
            continue
        window_metrics.append({
            "start_time": window[0]["timestamp"].isoformat(),
            "end_time": window[-1]["timestamp"].isoformat(),
            "message_count": len(window),
            "mean_similarity": float(np.mean([m["similarity"] for m in window])),
            "std_similarity": float(np.std([m["similarity"] for m in window])),
            "mean_distance": float(np.mean([m["distance"] for m in window])),
        })
    
    # Overall temporal stats
    similarities = [m["similarity"] for m in message_data]
    first_half = similarities[:len(similarities)//2]
    second_half = similarities[len(similarities)//2:]
    
    return {
        "time_span_days": total_span / 86400 if total_span > 0 else 0,
        "first_message": message_data[0]["timestamp"].isoformat(),
        "last_message": message_data[-1]["timestamp"].isoformat(),
        "windows": window_metrics,
        "evolution": {
            "early_mean_similarity": float(np.mean(first_half)) if first_half else None,
            "late_mean_similarity": float(np.mean(second_half)) if second_half else None,
            "similarity_trend": float(np.mean(second_half) - np.mean(first_half)) if first_half and second_half else None,
        },
        "stability": {
            "similarity_std": float(np.std(similarities)),
            "similarity_range": float(np.max(similarities) - np.min(similarities)),
        }
    }


def compute_identity_statistics(embeddings: np.ndarray, centroid: np.ndarray) -> Dict[str, float]:
    """Compute statistics about the identity embedding space."""
    # Distances from centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # Cosine similarities to centroid
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    similarities = np.dot(embeddings, centroid) / (norms + 1e-8)
    
    # Percentiles
    sorted_sims = np.sort(similarities)
    percentiles = {
        "p25": float(np.percentile(similarities, 25)),
        "p50": float(np.percentile(similarities, 50)),
        "p75": float(np.percentile(similarities, 75)),
        "p90": float(np.percentile(similarities, 90)),
        "p95": float(np.percentile(similarities, 95)),
    }
    
    return {
        "num_samples": len(embeddings),
        "embedding_dim": len(centroid),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "threshold_1std": float(np.mean(distances) + np.std(distances)),
        "threshold_2std": float(np.mean(distances) + 2 * np.std(distances)),
        "similarity_threshold_1std": float(np.mean(similarities) - np.std(similarities)),
        "similarity_threshold_2std": float(np.mean(similarities) - 2 * np.std(similarities)),
        "percentiles": percentiles,
    }


def save_identity_model(
    centroid: np.ndarray,
    stylistic_profile: Dict,
    vocabulary_profile: Dict,
    statistics: Dict,
    config: Dict,
    output_dir: Path,
    temporal_analysis: Optional[Dict] = None
):
    """Save the trained identity model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save centroid
    np.save(output_dir / "identity_centroid.npy", centroid)
    
    # Save profiles
    with open(output_dir / "stylistic_profile.json", 'w') as f:
        json.dump(stylistic_profile, f, indent=2)
    
    with open(output_dir / "vocabulary_profile.json", 'w') as f:
        json.dump(vocabulary_profile, f, indent=2)
    
    # Save temporal analysis if available
    if temporal_analysis:
        with open(output_dir / "temporal_analysis.json", 'w') as f:
            json.dump(temporal_analysis, f, indent=2, default=str)
    
    # Save config and statistics
    config_data = {
        **config,
        "statistics": statistics,
        "created_at": datetime.now().isoformat(),
        "centroid_shape": list(centroid.shape)
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n✅ Model saved to {output_dir}")


def verify_message(
    message: str,
    centroid: np.ndarray,
    model: SentenceTransformer,
    stylistic_profile: Dict,
    statistics: Dict
) -> Dict[str, Any]:
    """Verify a single message against the identity profile."""
    # Compute embedding
    embedding = model.encode([message], convert_to_numpy=True)[0]
    
    # Semantic similarity
    similarity = np.dot(embedding, centroid) / (
        np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
    )
    
    # Distance from centroid
    distance = np.linalg.norm(embedding - centroid)
    
    # Stylistic match
    msg_style = compute_stylistic_features(message)
    style_scores = []
    
    for feature, profile_stats in stylistic_profile.items():
        if feature in msg_style and "mean" in profile_stats:
            value = msg_style[feature]
            mean = profile_stats["mean"]
            std = profile_stats["std"] + 1e-8
            # Z-score
            z = abs(value - mean) / std
            # Convert to similarity (higher is better)
            style_scores.append(max(0, 1 - z / 3))  # 3 std = 0 similarity
    
    stylistic_match = np.mean(style_scores) if style_scores else 0.5
    
    # Combined score
    semantic_score = float(similarity)
    combined_score = 0.6 * semantic_score + 0.4 * stylistic_match
    
    # Determine confidence level
    sim_threshold = statistics.get("similarity_threshold_1std", 0.7)
    
    if semantic_score >= sim_threshold and stylistic_match >= 0.6:
        confidence = "high"
    elif semantic_score >= statistics.get("similarity_threshold_2std", 0.5):
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "semantic_similarity": semantic_score,
        "stylistic_match": float(stylistic_match),
        "combined_score": float(combined_score),
        "distance_from_centroid": float(distance),
        "confidence": confidence,
        "thresholds": {
            "similarity_1std": statistics.get("similarity_threshold_1std"),
            "similarity_2std": statistics.get("similarity_threshold_2std"),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train identity embedding model from conversation data"
    )
    parser.add_argument(
        "--model-size", choices=["tiny", "small", "base", "large"],
        default="base", help="Model size (default: base - 110M params)"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "auto"],
        default="auto", help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a test verification after training"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("IDENTITY EMBEDDING MODEL TRAINER")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Model: {ALTERNATIVE_MODELS[args.model_size]}")
    
    # Load conversations
    print("\n📚 Loading conversations...")
    conversations = load_conversations()
    
    if not conversations:
        print("No conversations found. Run parse_conversations.py first.")
        sys.exit(1)
    
    print(f"   Loaded {len(conversations)} conversations")
    
    # Load memory files for enhanced identity training
    print("\n🧠 Loading processed memory files...")
    memory_data = load_memory_files()
    identity_signals = extract_identity_signals(memory_data)
    
    print(f"   Identity signals extracted:")
    print(f"     • Distinctive terms: {len(identity_signals.get('distinctive_terms', []))}")
    print(f"     • Key topics: {len(identity_signals.get('key_topics', []))}")
    print(f"     • Key entities: {len(identity_signals.get('key_entities', []))}")
    print(f"     • Identity phrases: {len(identity_signals.get('identity_phrases', []))}")
    print(f"     • Naming events: {len(identity_signals.get('naming_events', []))}")
    
    # Extract user messages
    print("\n👤 Extracting user messages...")
    user_messages = extract_user_messages(conversations)
    print(f"   Found {len(user_messages)} user messages")
    
    if len(user_messages) < 50:
        print("\n⚠️  Warning: Less than 50 messages. Model may not be reliable.")
    
    # Load sentence transformer
    print("\n🤖 Loading sentence transformer model...")
    model_name = ALTERNATIVE_MODELS[args.model_size]
    model = SentenceTransformer(model_name, device=device)
    print(f"   Model loaded: {model_name}")
    
    # Compute embeddings
    print("\n🧮 Computing semantic embeddings...")
    embeddings = compute_semantic_embeddings(
        user_messages, model, batch_size=args.batch_size
    )
    
    # Compute identity centroid
    print("\n📍 Computing identity centroid...")
    centroid = compute_identity_centroid(embeddings)
    print(f"   Centroid shape: {centroid.shape}")
    
    # Compute statistics
    print("\n📊 Computing identity statistics...")
    statistics = compute_identity_statistics(embeddings, centroid)
    print(f"   Mean similarity to centroid: {statistics['mean_similarity']:.4f}")
    print(f"   Std similarity: {statistics['std_similarity']:.4f}")
    
    # Compute temporal analysis
    print("\n📅 Computing temporal evolution...")
    temporal_analysis = compute_temporal_analysis(user_messages, embeddings, centroid)
    if temporal_analysis:
        print(f"   Time span: {temporal_analysis.get('time_span_days', 0):.1f} days")
        print(f"   Time windows: {len(temporal_analysis.get('windows', []))}")
        if temporal_analysis.get('evolution', {}).get('similarity_trend'):
            trend = temporal_analysis['evolution']['similarity_trend']
            print(f"   Similarity trend: {trend:+.4f} (early → late)")
    else:
        print("   ⚠️  Insufficient temporal data for evolution analysis")
    
    # Compute stylistic profile
    print("\n✍️  Computing stylistic profile...")
    stylistic_profile = compute_stylistic_profile(user_messages)
    print(f"   Captured {len(stylistic_profile)} stylistic features")
    
    # Compute vocabulary fingerprint (enhanced with identity signals)
    print("\n📖 Computing vocabulary fingerprint...")
    vocabulary_profile = compute_vocabulary_fingerprint(user_messages, identity_signals)
    print(f"   Vocabulary size: {vocabulary_profile['vocabulary_size']}")
    print(f"   Distinctive terms: {len(vocabulary_profile.get('distinctive_terms', {}))}")
    print(f"   Identity-boosted terms: {len(vocabulary_profile.get('identity_boosted_terms', {}))}")
    
    # Save model
    print("\n💾 Saving identity model...")
    config = {
        "model_name": model_name,
        "model_size": args.model_size,
        "device_trained_on": device,
        "num_messages": len(user_messages),
        "num_conversations": len(conversations),
        "identity_signals": {
            "distinctive_terms_count": len(identity_signals.get("distinctive_terms", [])),
            "key_topics_count": len(identity_signals.get("key_topics", [])),
            "key_entities_count": len(identity_signals.get("key_entities", [])),
            "identity_phrases_count": len(identity_signals.get("identity_phrases", [])),
            "naming_events_count": len(identity_signals.get("naming_events", [])),
            "relational_markers": identity_signals.get("relational_markers", {}),
        }
    }
    
    save_identity_model(
        centroid=centroid,
        stylistic_profile=stylistic_profile,
        vocabulary_profile=vocabulary_profile,
        statistics=statistics,
        config=config,
        output_dir=MODELS_DIR,
        temporal_analysis=temporal_analysis if temporal_analysis else None
    )
    
    # Test verification
    if args.test and user_messages:
        print("\n🧪 Testing verification...")
        
        # Test with a real user message
        test_msg = user_messages[-1]["content"]
        result = verify_message(
            test_msg[:500],  # Truncate for display
            centroid, model, stylistic_profile, statistics
        )
        
        print(f"\n   Test message (truncated): {test_msg[:100]}...")
        print(f"   Semantic similarity: {result['semantic_similarity']:.4f}")
        print(f"   Stylistic match: {result['stylistic_match']:.4f}")
        print(f"   Combined score: {result['combined_score']:.4f}")
        print(f"   Confidence: {result['confidence']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📁 Model saved to: {MODELS_DIR}")
    print(f"\n📊 Identity Statistics:")
    print(f"   • Messages analyzed: {len(user_messages)}")
    print(f"   • Embedding dimension: {statistics['embedding_dim']}")
    print(f"   • Mean similarity: {statistics['mean_similarity']:.4f}")
    print(f"   • Verification threshold (1σ): {statistics['similarity_threshold_1std']:.4f}")
    print(f"   • Verification threshold (2σ): {statistics['similarity_threshold_2std']:.4f}")
    
    print("\n🔧 Next steps:")
    print("   1. Build the MCP: npm run build")
    print("   2. Use identity_verify tool to verify messages")
    print("   3. Integrate with identity providers via API")


if __name__ == "__main__":
    main()

