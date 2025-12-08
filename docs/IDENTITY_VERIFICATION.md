# Identity Verification System

A behavioral biometric system that creates an "identity fingerprint" from your conversation patterns, enabling verification of whether new messages match your identity.

## Overview

Traditional authentication verifies **what you have** (password, key) or **what you are** (fingerprint, face). This system verifies **how you communicate** — your unique patterns of expression that are difficult to fake.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOW IT WORKS                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Your Conversations                                             │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────┐                                           │
│   │ Train Identity  │  ◄── Learns your patterns                 │
│   │     Model       │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Identity Vector │  ◄── Your unique "fingerprint"            │
│   │   (Centroid)    │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Verify Messages │  ◄── Compare new input to fingerprint     │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Confidence Score: high / medium / low / none                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## System Requirements

### Training (GPU Recommended)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA 11.8+)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~500MB for model files
- **Python**: 3.9+

### Inference (CPU OK)
- **RAM**: 4GB minimum
- **Disk**: ~500MB for model files
- Works on CPU, faster with GPU

## Installation

```bash
# Install Python dependencies
pip install torch sentence-transformers numpy scikit-learn tqdm

# For GPU support (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For the verification service
pip install flask flask-cors
```

## Quick Start

### 1. Train Your Identity Model

```bash
cd scripts/identity_model
python train_identity_model.py --test
```

This will:
- Load your parsed conversations
- Extract user messages
- Compute semantic embeddings (what you talk about)
- Compute stylistic profile (how you write)
- Compute vocabulary fingerprint (what words you use)
- Save the identity model to `models/identity/`

### 2. Use MCP Tools

```bash
# Build and start the MCP
npm run build
npm start
```

Then use the verification tools:

```json
// Check model status
{ "tool": "identity_model_status" }

// Verify a message
{
  "tool": "identity_verify",
  "message": "Hey, I was thinking about that project we discussed..."
}

// Verify multiple messages
{
  "tool": "identity_verify_conversation",
  "messages": [
    "First message...",
    "Second message...",
    "Third message..."
  ]
}
```

### 3. (Optional) Run Full Semantic Service

For higher accuracy verification with deep semantic analysis, start the Python service:

**Option A: Direct**
```bash
pip install -r identity_service/requirements.txt
python identity_service/identity_service.py --port 4001
```

**Option B: Docker**
```bash
docker-compose --profile identity up -d identity-service
```

When the semantic service is running, the MCP automatically uses it for enhanced verification. Without it, MCP falls back to stylistic-only analysis (still useful, but ~70% as accurate).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP + Identity Service                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────┐    ┌────────────────────────────┐ │
│  │       MCP Server         │    │   Identity Service         │ │
│  │    (TypeScript/Node)     │◄──▶│   (Python + PyTorch)       │ │
│  │                          │    │                            │ │
│  │  • Stylistic analysis    │    │  • Sentence Transformers   │ │
│  │  • Vocabulary matching   │    │  • Semantic embeddings     │ │
│  │  • Fast, always-on       │    │  • GPU-accelerated         │ │
│  └──────────────────────────┘    └────────────────────────────┘ │
│                                                                  │
│  When semantic service is available:                            │
│    Combined Score = 0.6×Semantic + 0.25×Stylistic + 0.15×Vocab  │
│                                                                  │
│  When semantic service is unavailable (fallback):               │
│    Combined Score = 0.6×Stylistic + 0.4×Vocabulary              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## What Gets Analyzed

### Semantic Patterns (via Sentence Transformer)
- Topics and concepts you discuss
- How you frame ideas
- Semantic relationships in your language

### Stylistic Markers
- Punctuation habits (?, !, ..., —)
- Sentence and word length
- Capitalization patterns
- Use of code blocks, lists, quotes

### Vocabulary Fingerprint
- Distinctive words you use frequently
- Phrase patterns (bigrams)
- TF-IDF weighted distinctive terms

### Linguistic Markers
- First person singular vs plural (I vs we)
- Hedging words (maybe, perhaps, possibly)
- Certainty words (definitely, always, never)

## API Reference

### MCP Tools (4 for verification)

| Tool | Description |
|------|-------------|
| `identity_model_status` | Check if model is trained and ready |
| `identity_verify` | Verify a single message (auto-uses semantic if available) |
| `identity_verify_conversation` | Verify multiple messages |
| `identity_profile_summary` | Inspect the trained profile |

The MCP tools automatically detect if the Python semantic service is running. Response includes:
- `semantic_available: true/false` — whether semantic analysis was used
- `semantic_score: number` — semantic similarity (only when service is running)

### Python Verification Service (Optional, for full accuracy)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/status` | GET | Model status and statistics |
| `/verify` | POST | Verify single message |
| `/verify-batch` | POST | Verify multiple messages |

#### Example: Verify Message

```bash
curl -X POST http://localhost:4001/verify \
  -H "Content-Type: application/json" \
  -d '{"message": "Your message to verify..."}'
```

Response:
```json
{
  "available": true,
  "verified": true,
  "confidence": "high",
  "scores": {
    "semantic_similarity": 0.87,
    "stylistic_match": 0.82,
    "combined_score": 0.85,
    "distance_from_centroid": 0.34
  },
  "thresholds": {
    "high_confidence": 0.75,
    "medium_confidence": 0.55
  }
}
```

## Confidence Levels

| Level | Score Range | Interpretation |
|-------|-------------|----------------|
| **high** | ≥ 0.70 | Strong match to identity profile |
| **medium** | 0.50 - 0.69 | Likely match, some deviation |
| **low** | 0.30 - 0.49 | Weak match, significant deviation |
| **none** | < 0.30 | No match to identity profile |

## Model Files

After training, `models/identity/` contains:

```
models/identity/
├── config.json           # Model configuration and statistics
├── identity_centroid.npy # Identity embedding vector
├── stylistic_profile.json # Stylistic feature statistics
└── vocabulary_profile.json # Word frequency fingerprint
```

## Integration with Identity Providers

The verification system outputs a confidence score that can be used as a signal in identity platforms like Strata Maverics:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  User Message    │────▶│ Identity MCP     │────▶│ Identity Provider│
│                  │     │ Verification     │     │ (Strata, etc.)   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                              │
                              ▼
                    Confidence Score + Details
```

### Use Cases

1. **Continuous Authentication**: Verify identity throughout a session
2. **Anomaly Detection**: Flag messages that don't match the profile
3. **Multi-Factor**: Add behavioral biometrics to existing auth
4. **Fraud Prevention**: Detect account takeover attempts

## Training Options

```bash
# Use smaller model (faster, less accurate)
python train_identity_model.py --model-size tiny

# Use CPU only
python train_identity_model.py --device cpu

# Larger batch size (faster if you have GPU memory)
python train_identity_model.py --batch-size 64

# Run test verification after training
python train_identity_model.py --test
```

## Privacy & Security

- **All processing is local** — no data leaves your machine
- **You own your identity model** — it's just files on disk
- **Portable** — move the `models/identity/` folder anywhere
- **No cloud dependencies** — works completely offline

## Limitations

1. **Training data needed**: Requires ~50+ messages for reliable patterns
2. **Style can change**: Major life events may shift communication patterns
3. **Context matters**: Professional vs casual contexts have different patterns
4. **Not foolproof**: Sophisticated mimicry might pass low thresholds

## Docker Deployment

The identity service integrates with the main docker-compose:

```bash
# Start MCP only (stylistic verification)
docker-compose up -d mcp-server

# Start with full semantic verification
docker-compose --profile identity up -d

# Check if identity service is running
curl http://localhost:4001/health
```

The MCP server automatically connects to the identity service when available:
- `IDENTITY_SERVICE_URL=http://identity-service:4001` (set in docker-compose)

## Troubleshooting

### "Model not found"
Run `python train_identity_model.py` first.

### "CUDA out of memory"
Use `--model-size tiny` or `--device cpu`.

### Low confidence scores
- Need more training data (messages)
- Try re-training with `--model-size base`
- Check if testing with different context (formal vs casual)

### Slow inference
- Use GPU if available
- Use `--model-size tiny` for speed
- MCP's stylistic-only verification is faster than full semantic

### "semantic_available: false" in response
The Python service isn't running. Start it with:
```bash
python identity_service/identity_service.py --port 4001
# or
docker-compose --profile identity up -d identity-service
```

