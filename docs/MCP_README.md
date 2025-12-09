# Conversation Memory MCP — API Reference

An MCP (Model Context Protocol) server that provides a portable memory and identity layer for AI conversations. Implements the full MCP protocol (2024-11-05) with streaming support.

> **New here?** Start with the **[Getting Started Guide](./GETTING_STARTED.md)** to set up with your own ChatGPT data.

## Core Philosophy

> Identity is not data. Identity is the pattern of continuity—the unique rhythm, ordering, and interpretation of experiences through time.

This system preserves **patterns of continuity** from your conversations—providing access to memories, themes, symbols, and knowledge that emerged over time.

## Features

- **Model-Agnostic**: Works with any LLM (OpenAI, Anthropic, Ollama, local models, etc.)
- **Pattern Discovery**: Analyze your conversations to discover unique patterns and themes
- **MCP Protocol**: Full MCP 2024-11-05 implementation with streaming support
- **Comprehensive Access**: 39 tools for accessing memories, conversations, files, and identity verification
- **Fine-Tuning Support**: LoRA fine-tuning tools to train models on your conversation patterns
- **Filesystem-Backed**: JSONL files as canonical store for portability

## Quick Start

### Docker (Recommended)

```bash
docker-compose up --build
```

Services:
- **MCP Server**: `http://localhost:4000`
- **Ollama**: `http://localhost:11434`
- **LibreChat**: `http://localhost:3080`

### Local Development

```bash
npm install
npm run build
npm start
```

Default configuration:
- `PORT=4000`
- `MEMORY_DIR=./memory`

## Architecture

### Project Structure

```
identity-mcp/
├── src/
│   ├── index.ts              # Express server entrypoint
│   ├── config.ts             # Configuration loader
│   ├── routes/
│   │   ├── health.ts         # Health check endpoint
│   │   ├── httpApi.ts        # Direct HTTP REST API
│   │   └── mcpProtocol.ts    # MCP protocol implementation
│   ├── mcp/                  # MCP tool handlers
│   │   ├── memoryTools.ts
│   │   ├── memorySearchTools.ts
│   │   ├── identityTools.ts
│   │   ├── identityVerificationTools.ts
│   │   ├── identityAnalysisTools.ts
│   │   ├── emergenceTools.ts
│   │   ├── fileTools.ts
│   │   ├── conversationTools.ts
│   │   ├── statisticsTools.ts
│   │   ├── unifiedSearchTools.ts
│   │   ├── exportTools.ts
│   │   ├── finetuneTools.ts
│   │   └── memoryParserTools.ts
│   └── services/             # Core business logic
│       ├── fileStore.ts
│       ├── fileLoader.ts
│       ├── conversationLoader.ts
│       └── memoryParser.ts
├── memory/                   # JSONL memory files
├── conversations/            # Conversation history (JSONL files)
├── files/                   # RAG-able files (documents, notes, etc.)
├── scripts/
│   └── conversation_processing/  # Data processing scripts
│       ├── analyze_patterns.py
│       ├── parse_memories.py
│       ├── extract_conversations.py
│       ├── build_emergence_map.py
│       └── finetune_lora.py
└── training_data/           # Generated training datasets
```

### API Interfaces

1. **MCP Protocol** (`/mcp-protocol`): Full MCP 2024-11-05 with streaming
   - For MCP-compatible clients (LibreChat, Claude Desktop, etc.)
   - Uses JSON-RPC over HTTP/SSE

2. **Direct HTTP API** (`/mcp/*`): REST-style endpoints
   - For curl, Postman, scripts
   - Same tools, simpler interface

3. **Health Check** (`/health`): Server status and uptime

## MCP Tools Reference (39 Tools)

### Memory Tools (5 tools)

- **`memory_list`**: List memory files and record counts
- **`memory_get`**: Retrieve records from a memory file (with filters: type, tags, date range)
- **`memory_search`**: Full-text search across all memory files
- **`memory_append`**: Append a record to a memory file
- **`memory_parse`**: Rebuild user.context.jsonl from memories.json

### Identity Tools (2 tools)

- **`identity_get_core`**: Retrieve core identity patterns
- **`identity_get_full`**: Complete identity bundle (all memory files)

### Identity Analysis Tools (5 tools) — from analyze_identity.py

- **`identity_analysis_summary`**: Overview of identity pattern analysis
- **`identity_get_momentum`**: Patterns rising/falling over time (identity evolution)
- **`identity_get_naming_events`**: Moments where names/identities were established
- **`identity_get_clusters`**: Co-occurrence clusters (concepts that appear together)
- **`identity_get_relational`**: We/I ratios and role language patterns

### Emergence Tools (5 tools) — from build_emergence_map.py

- **`emergence_summary`**: Summary of emergence data (event counts, symbolic stats)
- **`emergence_get_events`**: Key events (naming, emotional, identity prompts)
- **`emergence_search`**: Search conversations by pattern/keyword/entity
- **`emergence_symbolic_conversations`**: Conversations with highest symbolic density
- **`emergence_timeline`**: Timeline of key events by date range

### File RAG Tools (4 tools)

- **`file_list`**: List files from RAG folders
- **`file_get`**: Retrieve a specific file by path
- **`file_search`**: Full-text search across files
- **`file_get_numbered`**: Get numbered files from a folder (by range or max count)

### Conversation Tools (4 tools)

- **`conversation_list`**: List all conversations with metadata
- **`conversation_get`**: Get a specific conversation by ID
- **`conversation_search`**: Search conversations by content
- **`conversation_by_date_range`**: Get conversations within a date range

### Statistics Tools (2 tools)

- **`memory_stats`**: Statistics about memory files (counts, types, tags, date ranges)
- **`conversation_stats`**: Statistics about conversations (total, messages, by year)

### Search Tools (1 tool)

- **`search_all`**: Unified search across memories, files, and conversations

### Export Tools (2 tools)

- **`export_memories`**: Export memories to JSONL or JSON
- **`export_conversations`**: Export conversations to JSONL or JSON

### Fine-Tuning Tools (5 tools)

- **`finetune_start`**: Start LoRA fine-tuning job
- **`finetune_status`**: Check fine-tuning job status
- **`finetune_list`**: List all fine-tuning jobs
- **`finetune_cancel`**: Cancel a running job
- **`finetune_export_dataset`**: Export training dataset without training

## Memory Files

The system uses JSONL files in `memory/`. Common files include:

- `identity.jsonl` - Core identity patterns (generated by analyze_patterns.py)
- `patterns.jsonl` - Keywords, topics, entities (generated, used by other scripts)
- `user.context.jsonl` - User context (generated from memories.json)

Any `.jsonl` file in the `memory/` directory is automatically loaded.

## Integration Examples

### LibreChat

Configure in `librechat.yaml`:
```yaml
mcpServers:
  memory-mcp:
    type: http
    url: http://mcp-server:4000/mcp-protocol
    timeout: 120000
```

### Direct HTTP

```bash
# Get core identity
curl -X POST http://localhost:4000/mcp/identity.get_core \
  -H "Content-Type: application/json" \
  -d '{}'

# Search memories
curl -X POST http://localhost:4000/mcp/memory.search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search term"}'

# Get conversations from date range
curl -X POST http://localhost:4000/mcp/conversation.by_date_range \
  -H "Content-Type: application/json" \
  -d '{"startDate": "2024-01-01", "endDate": "2024-12-31"}'

# Unified search
curl -X POST http://localhost:4000/mcp/search.all \
  -H "Content-Type: application/json" \
  -d '{"query": "topic"}'
```

### MCP Protocol (JSON-RPC)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "identity_get_core",
    "arguments": {}
  }
}
```

## Configuration

Environment variables:
- `PORT` - Server port (default: 4000)
- `MEMORY_DIR` - Memory files directory (default: ./memory)

Directories:
- `memory/` - Memory JSONL files
- `conversations/` - Conversation history
- `files/` - RAG-able files
- `training_data/` - Generated training datasets
- `adapters/` - Trained LoRA adapters

## Development

```bash
# Install dependencies
npm install

# Development mode (with hot reload)
npm run dev

# Build
npm run build

# Production
npm start
```

## Documentation
- **[GETTING_STARTED.md](./GETTING_STARTED.md)** - End-to-end setup with your ChatGPT data
- **[Identity Verification](./IDENTITY_VERIFICATION.md)** - Full Identity Verification documentation
- **[DOCKER_SETUP.md](./DOCKER_SETUP.md)** - Docker deployment guide


## Philosophy

This system embodies the core insight: **Identity is pattern, not data**.

The MCP serves as a searchable archive of your conversation history and patterns. It captures not just what was said, but how it was said—the recurring themes, language patterns, and continuity across conversations.

The MCP server is a **pattern access system** - making your complete conversation corpus accessible to any LLM that can call these tools.

## License

MIT
