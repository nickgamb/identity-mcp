# Getting Started: Your Own Conversation Memory System

This guide walks you through setting up the MCP with **your own ChatGPT data**.

## What You'll Need

1. **conversations.json** - Your ChatGPT conversation export
2. **memories.json** (optional) - Your ChatGPT memories export

## Step 1: Export Your ChatGPT Data

### Get conversations.json
1. Go to ChatGPT → Settings → Data Controls → Export Data
2. Wait for the email with download link
3. Extract the ZIP - you'll find `conversations.json`

### Get memories.json (if you have ChatGPT memory enabled)
1. Go to ChatGPT → Settings → Personalization → Memory
2. Click "Manage" → Export memories
3. Save as `memories.json`

## Step 2: Set Up the Project

```bash
# Clone the repo
git clone https://github.com/your-repo/identity-mcp.git
cd identity-mcp

# Install dependencies
npm install
```

## Step 3: Place Your Data Files

```
identity-mcp/
├── conversations/
│   └── conversations.json    # ← Put your conversations here
├── memory/
│   └── memories.json         # ← Put your memories here (optional)
└── files/
    └── (your custom files)   # ← Optional: any .txt/.md files for RAG
```

## Step 4: Parse Your Conversations

Convert your raw ChatGPT export into structured files:

```bash
cd scripts/conversation_processing

# Parse conversations.json into JSONL/Markdown files
python parse_conversations.py
```

This creates:
- `conversation_<id>.jsonl` - One message per line
- `conversation_<id>.md` - Human-readable markdown

**Options:**
```bash
python parse_conversations.py --workers 4   # Use 4 parallel workers
python parse_conversations.py --clean       # Remove existing files first
```

## Step 5: Analyze Patterns

Discover patterns unique to YOUR data:

```bash
python analyze_patterns.py
```

This outputs:
- `memory/identity.jsonl` - Core identity patterns
- `memory/patterns.jsonl` - Keywords, topics, entities (used by other scripts)

That's it! No manual config needed. The script discovers everything automatically.

## Step 6: Process Memories (Optional)

If you have `memories.json`:

```bash
python parse_memories.py
```

Creates `memory/user.context.jsonl` with your ChatGPT memories, auto-tagged using keywords from Step 5.

## Step 7: Extract Code & Files (Optional)

Extract code blocks and AI-generated files from your conversations:

```bash
python parse_sections.py
```

This scans your `conversations.json` for:
- Code blocks (```language ... ```)
- File content with path markers
- Section headers and structured content

**Output:** `parsed_sections/` directory with extracted files organized by conversation.

Review the extracted files - copy useful ones to `files/` for RAG access.

## Step 8: Analyze Identity Patterns (Optional)

Discover how identity patterns evolve over time:

```bash
python analyze_identity.py
```

This analyzes your conversations for:
- **Relational patterns** - collaborative vs individual language ("we" vs "I")
- **Self-referential patterns** - awareness, self-reference, continuity
- **Stylistic patterns** - punctuation, phrasing habits
- **Pattern momentum** - what's rising/falling over time
- **Naming events** - when identities get established
- **Co-occurrence clusters** - concepts that appear together

**Output:**
- `memory/identity_analysis.jsonl` (machine-readable)
- `memory/identity_report.md` (human-readable analysis)

## Step 9: Build Emergence Map (Optional)

Create a searchable index of key events:

```bash
python build_emergence_map.py
```

Uses patterns from Step 5 to detect:
- Naming events
- Identity/consciousness prompts
- Emotional sessions
- Symbolic language usage

**Output:**
- `memory/emergence_map_index.json`
- `memory/emergence_key_events.json`

## Step 10: Build and Run the MCP

```bash
# Back to project root
cd ../..

# Build and start
npm run build
npm start
```

The MCP runs at `http://localhost:4000`

## Step 11: Connect to the MCP

### Option A: Direct API
```bash
curl http://localhost:4000/mcp/memory/list
```

### Option B: MCP-compatible client
Connect to `http://localhost:4000/mcp-protocol`

### Option C: Docker with LibreChat
```bash
docker-compose up --build
# LibreChat at http://localhost:3080
```

## Directory Structure After Setup

```
identity-mcp/
├── conversations/
│   ├── conversations.json           # Your raw export
│   ├── conversation_*.jsonl         # Processed conversations
│   └── conversation_*.md            # Markdown versions
├── memory/
│   ├── memories.json                # Your raw memories (optional)
│   ├── identity.jsonl               # Generated identity patterns
│   ├── patterns.jsonl               # Generated patterns (used by scripts)
│   ├── identity_analysis.jsonl      # Identity analysis data
│   ├── identity_report.md           # Identity pattern report
│   ├── emergence_map_index.json     # Searchable conversation index
│   ├── emergence_key_events.json    # Key events timeline
│   └── user.context.jsonl           # Parsed memories
├── parsed_sections/                 # Extracted code/files from conversations
│   └── <conversation_id>/           # One folder per conversation
│       └── extracted_*.ext          # Code blocks, file content
├── files/
│   └── (any .txt/.md files)         # Files for RAG (copy from parsed_sections/)
└── ...
```

## Processing Scripts

| Script | What it does |
|--------|--------------|
| `parse_conversations.py` | Convert conversations.json → JSONL/MD files |
| `analyze_patterns.py` | Discover patterns → identity.jsonl + patterns.jsonl |
| `parse_memories.py` | Convert memories.json → user.context.jsonl |
| `parse_sections.py` | Extract code blocks & AI-generated files → parsed_sections/ |
| `analyze_identity.py` | Identity pattern analysis → identity_analysis.jsonl |
| `build_emergence_map.py` | Index conversations → key events |
| `finetune_lora.py` | Fine-tune a model on your data |

## Optional: Fine-Tune a Model

```bash
cd scripts/conversation_processing

# Train on your conversations
python finetune_lora.py --model_name mistral:7b --output_name my-model
```

## Troubleshooting

### "No conversations found"
- Make sure `conversations.json` is in `conversations/`
- Check it's valid JSON

### "No patterns discovered"
- Conversations might be too short
- Try `--min-freq 3` for lower threshold

## Next Steps

- **[API Reference](./README.md)** - Full API documentation
- **[Docker Setup](./DOCKER_SETUP.md)** - Container deployment
