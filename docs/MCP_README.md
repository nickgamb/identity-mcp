# Conversation Memory MCP — API Reference

An MCP (Model Context Protocol) server that provides a portable memory and identity layer for AI conversations. Implements the full MCP protocol (2024-11-05) with streaming support.

> **New here?** Start with the **[Getting Started Guide](./GETTING_STARTED.md)** to set up with your own ChatGPT data.

## Core Philosophy

> Identity is not data. Identity is the pattern of continuity—the unique rhythm, ordering, and interpretation of experiences through time.

This system preserves **patterns of continuity** from your conversations—providing access to memories, themes, symbols, and knowledge that developed over time.

## Features

- **Model-Agnostic**: Works with any LLM (OpenAI, Anthropic, Ollama, local models, etc.)
- **Pattern Discovery**: Analyze your conversations to discover unique patterns and themes
- **MCP Protocol**: Full MCP 2024-11-05 implementation with streaming support
- **Comprehensive Access**: 76 tools for memories, conversations, files, identity verification, semantic search, Letta agent, and reverie
- **Fine-Tuning Support**: LoRA fine-tuning with dual GPU support and CPU offloading
- **Letta Integration**: Stateful AI agent with archival memory (pgvector), core memory blocks, sleeptime agents
- **Reverie System**: Background self-reflection — the agent dream-walks its memories when the GPU is idle
- **Web Dashboard**: Upload, process, browse, and edit all data with Monaco editor
- **Pipeline Automation**: Run processing scripts via MCP tools or dashboard
- **Data Management**: Upload, clean, and inspect all source and generated data
- **Filesystem-Backed**: JSONL files as canonical store for portability

## Quick Start

### Docker (Recommended)

```bash
docker-compose up --build
```

Services:
- **MCP Server**: `http://localhost:4000`
- **Dashboard**: `http://localhost:3001`
- **Letta**: `http://localhost:8283`
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
│   ├── mcp/                  # MCP tool handlers (76 tools)
│   │   ├── memoryTools.ts
│   │   ├── memorySearchTools.ts
│   │   ├── identityTools.ts
│   │   ├── identityVerificationTools.ts
│   │   ├── identityAnalysisTools.ts
│   │   ├── interactionTools.ts
│   │   ├── fileTools.ts
│   │   ├── conversationTools.ts
│   │   ├── statisticsTools.ts
│   │   ├── unifiedSearchTools.ts
│   │   ├── semanticSearchTools.ts
│   │   ├── exportTools.ts
│   │   ├── finetuneTools.ts
│   │   ├── pipelineTools.ts
│   │   ├── dataManagementTools.ts
│   │   ├── eegIdentityTools.ts
│   │   ├── lettaProxy.ts        # Letta agent proxy (archival, core memory, config)
│   │   └── memoryParserTools.ts
│   ├── services/             # Core business logic
│   │   ├── fileStore.ts
│   │   ├── fileLoader.ts
│   │   ├── conversationLoader.ts
│   │   ├── reverieService.ts    # Background self-reflection loop
│   │   └── memoryParser.ts
│   └── utils/
│       └── reveriePrompts.ts    # Reverie prompt config (external JSON)
├── memory/                   # JSONL memory files
├── conversations/            # Conversation history (JSONL files)
├── files/                   # RAG-able files (documents, notes, etc.)
├── scripts/
│   └── conversation_processing/  # Data processing scripts
│       ├── analyze_patterns.py
│       ├── parse_memories.py
│       ├── extract_conversations.py
│       ├── build_interaction_map.py
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

## MCP Tools Reference (76 Tools)

### Memory Tools (7 tools)

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

### Interaction Map Tools (5 tools) — from build_interaction_map.py
Focus: Human communication patterns and identity fingerprinting

- **`interaction_summary`**: Summary of interaction data (event counts, topic/tone distribution, human message stats)
- **`interaction_get_events`**: Key human communication events (problem-solving, tempo changes, topic transitions, tone shifts)
- **`interaction_search`**: Search conversations by topic, tone, or keyword
- **`interaction_get_by_topic`**: Get conversations filtered by specific topic tag
- **`interaction_timeline`**: Timeline of key human communication events by date range

### File RAG Tools (6 tools)

- **`file_list`**: List files from RAG folders
- **`file_get`**: Retrieve a specific file by path
- **`file_search`**: Full-text search across files
- **`file_get_numbered`**: Get numbered files from a folder (by range or max count)
- **`file_upload`**: Upload a file to the RAG folder
- **`file_delete`**: Delete a file from the RAG folder

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

- **`finetune_start`**: Start LoRA fine-tuning job (supports CPU-only, single/multi-GPU)
- **`finetune_status`**: Check fine-tuning job status
- **`finetune_list`**: List all fine-tuning jobs
- **`finetune_cancel`**: Cancel a running job
- **`finetune_export_dataset`**: Export training dataset without training

### Pipeline Tools (6 tools)

- **`pipeline_list`**: List available processing scripts
- **`pipeline_run`**: Run a specific processing script
- **`pipeline_run_all`**: Run all scripts in order (stops on failure)
- **`pipeline_status`**: Check if a specific script is running
- **`pipeline_list_running`**: List all currently running scripts
- **`pipeline_stop`**: Stop a running pipeline script

### Data Management Tools (12 tools)

- **`data_status`**: Check presence of source files and generated data
- **`data_upload_conversations`**: Upload conversations.json (overwrites existing)
- **`data_upload_memories`**: Upload memories.json (overwrites existing)
- **`data_upload_anthropic_conversations`**: Upload Anthropic conversations export
- **`data_upload_anthropic_memories`**: Upload Anthropic memories export
- **`data_clean`**: Clean generated data from a directory (keeps source files)
- **`data_delete_source`**: Delete a specific source data file
- **`data_conversations_list`**: List all parsed conversation files with metadata
- **`data_conversation_get`**: Get specific conversation content by ID
- **`data_conversation_update`**: Update conversation content
- **`data_memories_list`**: List all memory records from all files
- **`data_memory_file_get`**: Get specific memory file content
- **`data_memory_file_update`**: Update memory file content

### Identity Verification Tools (4 tools)

- **`identity_model_status`**: Check if identity verification model is trained
- **`identity_verify`**: Verify a single message against identity profile
- **`identity_verify_conversation`**: Verify multiple messages
- **`identity_profile_summary`**: Get trained identity profile summary

### EEG Identity Assurance Tools (4 tools)

- **`eeg_model_status`**: Check EEG identity model training status
- **`eeg_enroll`**: Enroll EEG biometric data for identity verification
- **`eeg_authorize`**: Authorize a session via EEG biometric match
- **`eeg_profile_summary`**: Get EEG identity profile summary

### Semantic Search Tools (1 tool)

- **`search_semantic`**: Vector-based semantic search across archival memory (requires Letta + embedding model)

### Letta Agent Tools (7 tools)

- **`ollama_models`**: List models installed on the Ollama server
- **`letta_status`**: Get Letta agent status (agent info, sleeptime config, model handles)
- **`letta_memory`**: Get core memory blocks (persona, human)
- **`letta_memory_update`**: Update a core memory block
- **`letta_archival`**: Search or list archival memory passages (pgvector)
- **`letta_messages`**: Get recent agent messages (conversation + sleeptime activity)
- **`letta_config`**: Update Letta agent settings (sleeptime, model, embedding, timezone)

### Reverie Tools (4 tools)

- **`reverie_status`**: Get reverie status (running, last reverie time, next prompt, config)
- **`reverie_config`**: Update reverie settings (enabled, interval 30-720 min)
- **`reverie_prompts_get`**: Get the current list of self-reflection prompts
- **`reverie_prompts_update`**: Replace the reverie prompt list

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
- `LETTA_BASE_URL` - Letta server URL (default: `http://letta:8283`)
- `LETTA_AGENT_NAME` - Letta agent name (default: `identity`)
- `OLLAMA_BASE_URL` - Ollama server URL (default: `http://ollama:11434`)
- `REVERIE_ENABLED` - Enable background self-reflection (default: `false`)
- `REVERIE_INTERVAL_MINUTES` - Minutes between reveries (default: `120`, min 30)

See **[Environment Variables](./ENVIRONMENT_VARIABLES.md)** for the full reference.

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

## Related Documentation

- **[Getting Started](./GETTING_STARTED.md)** - End-to-end setup with your ChatGPT data
- **[Identity Verification](./IDENTITY_VERIFICATION.md)** - How the verification system works
- **[Multi-User & OIDC Support](./MULTI_USER_OIDC.md)** - Multi-user data isolation and OIDC authentication
- **[Docker Setup](./DOCKER_SETUP.md)** - Container deployment guide
- **[Environment Variables](./ENVIRONMENT_VARIABLES.md)** - Complete reference for all configuration options
- **[Blog: Securing Identity MCP](./BLOG_SECURING_IDENTITY_MCP.md)** - Tutorial on adding OAuth/OIDC and policy-based access control