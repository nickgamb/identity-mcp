#!/usr/bin/env python3
"""
================================================================================
IDENTITY VERIFICATION SERVICE
================================================================================

PURPOSE:
    HTTP service that provides full semantic identity verification using the
    trained sentence transformer model. Complements the MCP's stylistic
    verification with deep semantic analysis.

ENDPOINTS:
    POST /verify         - Verify a single message
    POST /verify-batch   - Verify multiple messages
    GET  /status         - Model status and statistics
    GET  /health         - Service health check

USAGE:
    python identity_service.py [--port 4001] [--host 0.0.0.0]

INTEGRATION:
    The MCP can call this service for full verification, or use its built-in
    stylistic verification for faster (but less accurate) checks.

================================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check for Flask
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Check for ML dependencies
try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_ML = True
except ImportError:
    HAS_ML = False

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASE_MODELS_DIR = PROJECT_ROOT / "models" / "identity"

# Global model storage (per-user cache)
models_cache: Dict[str, Dict] = {}  # userId -> {model, centroid, config, stylistic_profile}

def get_models_dir(user_id: Optional[str] = None) -> Path:
    """Get models directory for a user."""
    if user_id:
        return BASE_MODELS_DIR / user_id
    return BASE_MODELS_DIR

def load_model(user_id: Optional[str] = None):
    """Load the trained identity model for a user."""
    global models_cache
    
    # Check cache first
    cache_key = user_id or "default"
    if cache_key in models_cache:
        cached = models_cache[cache_key]
        return True, f"Model loaded from cache (user: {user_id or 'default'})"
    
    models_dir = get_models_dir(user_id)
    
    config_path = models_dir / "config.json"
    centroid_path = models_dir / "identity_centroid.npy"
    stylistic_path = models_dir / "stylistic_profile.json"
    
    if not config_path.exists():
        return False, f"Model not found for user {user_id or 'default'}. Run train_identity_model.py first."
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load centroid
    centroid = np.load(centroid_path)
    
    # Load stylistic profile
    with open(stylistic_path, 'r') as f:
        stylistic_profile = json.load(f)
    
    # Load sentence transformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config["model_name"], device=device)
    
    # Cache the loaded model
    models_cache[cache_key] = {
        "model": model,
        "centroid": centroid,
        "config": config,
        "stylistic_profile": stylistic_profile
    }
    
    return True, f"Model loaded on {device} (user: {user_id or 'default'})"


def compute_stylistic_features(text: str) -> Dict[str, float]:
    """Extract stylistic features (same as training)."""
    import re
    
    if not text:
        return {}
    
    char_count = len(text)
    word_count = len(text.split())
    
    return {
        "avg_word_length": sum(len(w) for w in text.split()) / max(word_count, 1),
        "question_marks": text.count("?") / char_count * 100,
        "exclamation_marks": text.count("!") / char_count * 100,
        "commas": text.count(",") / char_count * 100,
        "first_person_singular": len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / max(word_count, 1),
        "first_person_plural": len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.I)) / max(word_count, 1),
    }


def verify_message(message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Verify a message against the identity profile."""
    cache_key = user_id or "default"
    
    if cache_key not in models_cache:
        return {"error": f"Model not loaded for user {user_id or 'default'}"}
    
    cached = models_cache[cache_key]
    model = cached["model"]
    centroid = cached["centroid"]
    config = cached["config"]
    stylistic_profile = cached["stylistic_profile"]
    
    # Compute semantic embedding
    embedding = model.encode([message], convert_to_numpy=True)[0]
    
    # Semantic similarity (cosine)
    similarity = np.dot(embedding, centroid) / (
        np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8
    )
    
    # Distance from centroid
    distance = float(np.linalg.norm(embedding - centroid))
    
    # Stylistic match
    msg_style = compute_stylistic_features(message)
    style_scores = []
    
    for feature, profile_stats in stylistic_profile.items():
        if feature in msg_style and "mean" in profile_stats:
            value = msg_style[feature]
            mean = profile_stats["mean"]
            std = profile_stats["std"] + 1e-8
            z = abs(value - mean) / std
            style_scores.append(max(0, 1 - z / 3))
    
    stylistic_match = np.mean(style_scores) if style_scores else 0.5
    
    # Combined score
    semantic_score = float(similarity)
    combined_score = 0.7 * semantic_score + 0.3 * float(stylistic_match)
    
    # Determine confidence
    stats = config.get("statistics", {})
    sim_threshold_1std = stats.get("similarity_threshold_1std", 0.7)
    sim_threshold_2std = stats.get("similarity_threshold_2std", 0.5)
    
    if semantic_score >= sim_threshold_1std and stylistic_match >= 0.6:
        confidence = "high"
        verified = True
    elif semantic_score >= sim_threshold_2std:
        confidence = "medium"
        verified = True
    elif semantic_score >= sim_threshold_2std - 0.1:
        confidence = "low"
        verified = False
    else:
        confidence = "none"
        verified = False
    
    return {
        "verified": verified,
        "confidence": confidence,
        "scores": {
            "semantic_similarity": semantic_score,
            "stylistic_match": float(stylistic_match),
            "combined_score": combined_score,
            "distance_from_centroid": distance
        },
        "thresholds": {
            "high_confidence": sim_threshold_1std,
            "medium_confidence": sim_threshold_2std
        }
    }


def create_app():
    """Create Flask application."""
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health():
        user_id = request.args.get('user_id') or request.headers.get('X-User-Id')
        cache_key = user_id or "default"
        model_loaded = cache_key in models_cache
        
        return jsonify({
            "status": "healthy",
            "model_loaded": model_loaded,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/status', methods=['GET'])
    def status():
        user_id = request.args.get('user_id') or request.headers.get('X-User-Id')
        cache_key = user_id or "default"
        
        if cache_key not in models_cache:
            return jsonify({
                "available": False,
                "message": f"Model not loaded for user {user_id or 'default'}"
            })
        
        config = models_cache[cache_key]["config"]
        return jsonify({
            "available": True,
            "user_id": user_id,
            "model": config.get("model_name"),
            "model_size": config.get("model_size"),
            "messages_trained_on": config.get("num_messages"),
            "conversations": config.get("num_conversations"),
            "created_at": config.get("created_at"),
            "statistics": config.get("statistics", {})
        })
    
    @app.route('/verify', methods=['POST'])
    def verify():
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"success": False, "message": "Missing 'message' field"}), 400
        
        # Get user_id from request (query param, header, or body)
        user_id = data.get('user_id') or request.args.get('user_id') or request.headers.get('X-User-Id')
        cache_key = user_id or "default"
        
        if cache_key not in models_cache:
            # Try to load model if not cached
            success, message = load_model(user_id)
            if not success:
                return jsonify({
                    "success": False,
                    "message": message
                }), 503
        
        result = verify_message(data['message'], user_id)
        
        # Format response for MCP integration
        return jsonify({
            "success": True,
            "similarity": result["scores"]["semantic_similarity"],
            "confidence": result["confidence"],
            "threshold": result["thresholds"]["high_confidence"],
            "verified": result["verified"],
            "scores": result["scores"],
            "thresholds": result["thresholds"]
        })
    
    @app.route('/verify-batch', methods=['POST'])
    def verify_batch():
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({"error": "Missing 'messages' field"}), 400
        
        # Get user_id from request
        user_id = data.get('user_id') or request.args.get('user_id') or request.headers.get('X-User-Id')
        cache_key = user_id or "default"
        
        if cache_key not in models_cache:
            # Try to load model if not cached
            success, message = load_model(user_id)
            if not success:
                return jsonify({
                    "available": False,
                    "error": message
                }), 503
        
        messages = data['messages']
        results = []
        total_score = 0
        
        for msg in messages:
            result = verify_message(msg, user_id)
            results.append({
                "message_preview": msg[:50] + ("..." if len(msg) > 50 else ""),
                **result
            })
            total_score += result['scores']['combined_score']
        
        overall_score = total_score / len(messages) if messages else 0
        
        if overall_score >= 0.7:
            overall_confidence = "high"
            overall_verified = True
        elif overall_score >= 0.5:
            overall_confidence = "medium"
            overall_verified = True
        else:
            overall_confidence = "low"
            overall_verified = False
        
        return jsonify({
            "available": True,
            "overall_verified": overall_verified,
            "overall_confidence": overall_confidence,
            "overall_score": overall_score,
            "results": results
        })
    
    return app


def main():
    parser = argparse.ArgumentParser(description="Identity Verification Service")
    parser.add_argument("--port", type=int, default=4001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if not HAS_FLASK:
        print("Error: Flask not installed. Run: pip install flask flask-cors")
        sys.exit(1)
    
    if not HAS_ML:
        print("Error: ML dependencies not installed. Run: pip install torch sentence-transformers")
        sys.exit(1)
    
    print("=" * 60)
    print("IDENTITY VERIFICATION SERVICE")
    print("=" * 60)
    print("\n💡 Note: Models are loaded on-demand per user.")
    print("   Use ?user_id=<id> query param or X-User-Id header to specify user.")
    print("   Default user model will be loaded if no user_id provided.\n")
    
    # Start server
    app = create_app()
    
    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print(f"   GET  http://{args.host}:{args.port}/health")
    print(f"   GET  http://{args.host}:{args.port}/status")
    print(f"   POST http://{args.host}:{args.port}/verify")
    print(f"   POST http://{args.host}:{args.port}/verify-batch")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

