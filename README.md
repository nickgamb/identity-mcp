# Identity MCP

A behavioral identity verification system that creates an "identity fingerprint" from your conversation history. Train a model on how you communicate, then verify if new messages match your identity.

## What It Does

- **Parses** your ChatGPT formatted conversation export (conversations.json, memories.json)
- **Discovers** patterns unique to you (topics, vocabulary, style)
- **Trains** an embedding model on your communication patterns
- **Verifies** new messages against your identity fingerprint
- **Serves** identity data via MCP protocol (for LibreChat, etc.)

## Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### Chat Interface
![Chat](screenshots/chat.png)

### Docker Compose
![Docker Compose](screenshots/compose.png)

## Quick Start

```bash
# 1. Export your data from ChatGPT and place in project
cp ~/Downloads/conversations.json conversations/
cp ~/Downloads/memories.json memory/

# 2. Process your data
cd scripts/conversation_processing
python parse_conversations.py
python analyze_patterns.py
python parse_memories.py
python analyze_identity.py
python build_emergence_map.py

# 3. Train identity model
cd ../identity_model
python train_identity_model.py

# 4. Start services
cd ../..
docker-compose up -d                              # MCP only
docker-compose --profile identity up -d           # MCP + Identity Service
```

## Documentation

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Full setup guide with all options |
| [Identity Verification](docs/IDENTITY_VERIFICATION.md) | How the verification system works |
| [Docker Setup](docs/DOCKER_SETUP.md) | Container deployment guide |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RAW DATA (from ChatGPT export)                                         │
│  ├── conversations.json                                                  │
│  └── memories.json                                                       │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PROCESSING SCRIPTS (scripts/conversation_processing/)          │    │
│  │                                                                  │    │
│  │  1. parse_conversations.py                                       │    │
│  │     → conversations/*.jsonl (parsed messages)                    │    │
│  │     → conversations/*.md (human-readable)                        │    │
│  │                                                                  │    │
│  │  2. analyze_patterns.py                                          │    │
│  │     → memory/identity.jsonl (core identity patterns)             │    │
│  │     → memory/patterns.jsonl (keywords, topics, entities)         │    │
│  │                                                                  │    │
│  │  3. parse_memories.py                                            │    │
│  │     → memory/user.context.jsonl (ChatGPT memories as context)    │    │
│  │                                                                  │    │
│  │  4. analyze_identity.py                                          │    │
│  │     → memory/identity_analysis.jsonl (relational/stylistic)      │    │
│  │                                                                  │    │
│  │  5. build_emergence_map.py                                       │    │
│  │     → memory/emergence_map_index.json (searchable index)         │    │
│  │     → memory/emergence_key_events.json (significant moments)     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  IDENTITY MODEL TRAINING (scripts/identity_model/)              │    │
│  │                                                                  │    │
│  │  train_identity_model.py                                         │    │
│  │     READS:                                                       │    │
│  │       • conversations/*.jsonl (user messages)                    │    │
│  │       • memory/patterns.jsonl (boosts distinctive terms)         │    │
│  │       • memory/identity.jsonl (identity phrases)                 │    │
│  │       • memory/identity_analysis.jsonl (relational markers)      │    │
│  │       • memory/user.context.jsonl (ChatGPT memories)             │    │
│  │     OUTPUTS:                                                     │    │
│  │       models/identity/                                           │    │
│  │       ├── config.json (model info, thresholds, signals)          │    │
│  │       ├── identity_centroid.npy (semantic "fingerprint")         │    │
│  │       ├── stylistic_profile.json (how you write)                 │    │
│  │       └── vocabulary_profile.json (words + identity-boosted)     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  RUNTIME SERVICES                                                │    │
│  │                                                                  │    │
│  │  ┌──────────────────────┐    ┌────────────────────────────────┐ │    │
│  │  │  MCP Server (:4000)  │◄──▶│ Identity Service (:4001)       │ │    │
│  │  │                      │    │                                │ │    │
│  │  │ • memory/*.jsonl     │    │ • Loads trained model          │ │    │
│  │  │ • files/* (RAG)      │    │ • Sentence transformer         │ │    │
│  │  │ • conversations/*    │    │ • Computes similarity to       │ │    │
│  │  │ • Stylistic check    │    │   identity centroid            │ │    │
│  │  │   (fallback)         │    │                                │ │    │
│  │  └──────────────────────┘    └────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  VERIFICATION (during live conversation)                         │    │
│  │                                                                  │    │
│  │  User sends message → MCP calls identity_verify →                │    │
│  │    IF identity-service running:                                  │    │
│  │      60% semantic (distance from centroid)                       │    │
│  │      25% stylistic (punctuation, sentence length, etc)           │    │
│  │      15% vocabulary (distinctive words)                          │    │
│  │    ELSE fallback:                                                │    │
│  │      60% stylistic + 40% vocabulary                              │    │
│  │                                                                  │    │
│  │  Returns: { verified: true/false, confidence: high/medium/low }  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## MCP Tools (39 total)

The MCP server exposes tools for:
- **Memory** - Read/write/search identity memories
- **Conversations** - Query parsed conversation history  
- **Identity Analysis** - Relational patterns, naming events, momentum
- **Emergence** - Key events, symbolic density, timeline
- **Identity Verification** - Verify messages against your fingerprint
- **Files** - RAG over your documents
- **Fine-tuning** - Export datasets for model training

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for full tool reference.

## Requirements

- **Node.js 18+** (MCP server)
- **Python 3.9+** (processing scripts, identity service)
- **GPU recommended** for training (works on CPU, just slower)

## Roadmap

| Status | Feature |
|--------|---------|
| ✅ | Parse ChatGPT conversation exports |
| ✅ | Discover identity patterns (vocabulary, style, topics) |
| ✅ | Train semantic embedding model |
| ✅ | Identity verification via MCP tools |
| ✅ | Memory files enhance training (boosts distinctive terms) |
| 🔲 | **Non-conversational data support** - Train on essays, journals, emails, blog posts, social media |
| 🔲 | Multiple identity profiles (compare/switch between identities) |
| 🔲 | Identity drift detection (alert when patterns change over time) |
| 🔲 | Export identity model for use in other systems |

#License

Apache License 2.0
See the LICENSE.md file for full details.