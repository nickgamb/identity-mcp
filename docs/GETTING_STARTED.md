# Getting Started: Your Own Conversation Memory System

This guide walks you through setting up the MCP with **your own ChatGPT data**.

## What You'll Need

1. **conversations.json** - Your ChatGPT conversation export
2. **memories.json** (optional) - Your ChatGPT memories export

## Two Paths: Dashboard (GUI) or Command Line

### 🎨 Dashboard Path (Recommended for First-Time Setup)
- Upload files via web interface
- Run scripts with one click
- Monitor progress in real-time
- Browse, search, and edit all data
- Visual status indicators

### 💻 Command Line Path (For Automation)
- Script-driven workflow
- No GUI dependency
- Good for CI/CD or headless environments

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

# Install Python dependencies (for processing scripts)
pip install -r scripts/conversation_processing/requirements.txt  # If it exists
```

---

## 🎨 DASHBOARD PATH (Recommended)

### Step 3A: Start Services

```bash
# Terminal 1 - Start MCP server
npm start                    # Runs on http://localhost:4000

# Terminal 2 - Start Dashboard
cd dashboard
npm install
npm run dev                  # Dashboard on http://localhost:3001
```

### Step 4A: Upload Data via Dashboard

1. Open http://localhost:3001
2. Click **"Data Explorer"** button
3. Go to **"Status & Upload"** tab
4. Click **"Upload conversations.json"** and select your file
5. Click **"Upload memories.json"** if you have one

### Step 5A: Run Pipeline

1. Click **"Pipeline"** button in dashboard
2. Click **"Run"** on each script in order:
   - `parse_conversations` (parse raw data)
   - `analyze_patterns` (discover patterns)
   - `parse_memories` (if you uploaded memories.json)
   - `analyze_identity` (optional - identity analysis)
   - `build_interaction_map` (optional - human communication patterns)
   - `train_identity_model` (optional - verification model)

3. Watch real-time output in the right panel

### Step 6A: Browse Your Data

1. Go back to **"Data Explorer"**
2. Browse **"Conversations"** tab - search, view, edit any conversation
3. Browse **"Memories"** tab - view and edit all memory files
4. Browse **"Files"** tab - manage RAG files

**Done!** Your data is processed and accessible via MCP tools.

---

## 💻 COMMAND LINE PATH

### Step 3B: Place Your Data Files

```
identity-mcp/
├── conversations/
│   └── conversations.json    # ← Put your conversations here
├── memory/
│   └── memories.json         # ← Put your memories here (optional)
└── files/
    └── (your custom files)   # ← Optional: any .txt/.md files for RAG
```

### Step 4B: Parse Your Conversations

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

## Step 9: Build Interaction Map (Optional)

Create a searchable index of human communication patterns:

```bash
python build_interaction_map.py
```

Focus: Human identity fingerprinting - detects:
- Problem-solving moments (complex problem discussions)
- Communication tempo changes (sudden shifts in message length/cadence)
- Topic transitions (how you move between topics)
- Tone shifts (emotional pattern changes)

**Output:**
- `memory/interaction_map_index.json` (searchable conversation index with topics/tones)
- `memory/interaction_key_events.json` (key human communication events)

### Step 10B: Build and Run the MCP

```bash
# Back to project root
cd ../..

# Build and start
npm run build
npm start
```

The MCP runs at `http://localhost:4000`

---

## Connect to the MCP (Both Paths)

### Option A: Dashboard (Already running if you used Dashboard Path)
- Open http://localhost:3001
- Explore data, run scripts, edit content

### Option B: Direct API
```bash
curl http://localhost:4000/mcp/memory.list
curl http://localhost:4000/mcp/conversation.list
curl http://localhost:4000/mcp/identity.get_core
```

### Option C: MCP-compatible client
Connect to `http://localhost:4000/mcp-protocol`

### Option D: Docker with LibreChat
```bash
docker-compose up --build
# LibreChat at http://localhost:3080
# Dashboard at http://localhost:3001
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
│   ├── interaction_map_index.json     # Searchable conversation index
│   ├── interaction_key_events.json    # Key human communication events
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
| `build_interaction_map.py` | Index conversations → human communication patterns |
| `finetune_lora.py` | Fine-tune a model on your data |

## Optional: Fine-Tune a Model

Fine-tune a model on your conversation patterns:

```bash
cd scripts/conversation_processing

# CPU-only (slow, for testing)
python finetune_lora.py --model_name alden-gpt-oss-20b --output_name my-adapter

# Single GPU (20GB VRAM limit with CPU offload)
python finetune_lora.py --model_name alden-gpt-oss-20b --output_name my-adapter --max_vram_per_gpu 20

# Single GPU (no CPU offload - all on GPU)
python finetune_lora.py --model_name alden-gpt-oss-20b --output_name my-adapter --no_cpu_offload

# Multi-GPU with DeepSpeed (must use accelerate)
accelerate launch finetune_lora.py --model_name alden-gpt-oss-20b --output_name my-adapter --max_vram_per_gpu 20

# Multi-GPU with auto VRAM detection
accelerate launch finetune_lora.py --model_name alden-gpt-oss-20b --output_name my-adapter --max_vram_per_gpu auto
```

**Options:**
- `--max_vram_per_gpu N` - Limit VRAM to N GB (default 20)
- `--max_vram_per_gpu auto` - Use all available VRAM
- `--no_cpu_offload` - Disable CPU offloading (GPU-only mode)

See script header comments for more examples and configuration options.

## Troubleshooting

### "No conversations found"
- Make sure `conversations.json` is in `conversations/`
- Check it's valid JSON

### "No patterns discovered"
- Conversations might be too short
- Try `--min-freq 3` for lower threshold

## Documentation

- **[MCP Readme](./MCP_README.md)** - Full MCP documentation
- **[Identity Verification](./IDENTITY_VERIFICATION.md)** - Full Identity Verification documentation
- **[DOCKER_SETUP.md](./DOCKER_SETUP.md)** - Docker deployment guide
