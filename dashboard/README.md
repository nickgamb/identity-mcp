# Identity MCP Dashboard

A simple admin dashboard for managing the identity processing pipeline.

## Features

- **Script Management**: View and run all processing scripts
- **Pipeline Visualization**: See the processing flow at a glance
- **Output Viewer**: Real-time script output in terminal-style display
- **File Inspector**: View generated output files
- **MCP Status**: Monitor MCP server health

## Quick Start

```bash
# Install dependencies
npm install

# Start development server (connects to MCP on localhost:4000)
npm run dev
```

Open http://localhost:3001

## Prerequisites

The MCP server must be running on port 4000:

```bash
# From project root
docker-compose up -d mcp-server
# or
npm start
```

## Available Scripts

The dashboard can run these scripts via the MCP pipeline API:

| Script | Purpose |
|--------|---------|
| `parse_conversations` | Parse raw conversations.json → JSONL |
| `analyze_patterns` | Discover patterns → identity.jsonl, patterns.jsonl |
| `parse_memories` | Parse memories.json → user.context.jsonl |
| `analyze_identity` | Extract identity markers → identity_analysis.jsonl |
| `build_emergence_map` | Index conversations → key events |
| `train_identity_model` | Train embedding model → models/identity/ |

## Architecture

```
Dashboard (React/Vite :3001)
    │
    ├── /api/* ─────→ MCP Server (:4000)
    │                    │
    │                    ├── /mcp/pipeline.list
    │                    ├── /mcp/pipeline.run
    │                    └── /mcp/pipeline.run_all
    │
    └── Static assets
```

## Build for Production

```bash
npm run build
npm run preview
```

Output in `dist/` can be served by any static file server.

