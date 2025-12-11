# Identity MCP Dashboard

A comprehensive web interface for managing your identity data and processing pipeline.

## Features

### Pipeline Management
- **Script Management**: View and run all processing scripts
- **Pipeline Visualization**: See the processing flow at a glance
- **Output Viewer**: Real-time script output in terminal-style display
- **Status Tracking**: Monitor running scripts and completion status
- **MCP Status**: Live health monitoring with auto-refresh

### Data Management (NEW!)
- **Upload**: Drag & drop conversations.json and memories.json
- **Status Dashboard**: Visual indicators for all data sources
- **Data Explorer**: Browse, search, and edit all data
- **Monaco Editor**: VS Code-like editor for conversations and memories
- **Search**: Real-time filtering across all data
- **Clean**: Safely remove generated data (keeps source files)

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
| `build_interaction_map` | Index conversations → human communication patterns |
| `train_identity_model` | Train embedding model → models/identity/ |

## Architecture

```
Dashboard (React/Vite :3001)
    │
    ├── /api/* ─────→ MCP Server (:4000)
    │                    │
    │                    ├── Pipeline API
    │                    │   ├── /mcp/pipeline.list
    │                    │   ├── /mcp/pipeline.run
    │                    │   ├── /mcp/pipeline.status
    │                    │   └── /mcp/pipeline.run_all
    │                    │
    │                    ├── Data Management API (NEW)
    │                    │   ├── /mcp/data.status
    │                    │   ├── /mcp/data.upload_conversations
    │                    │   ├── /mcp/data.upload_memories
    │                    │   ├── /mcp/data.clean
    │                    │   ├── /mcp/data.conversations
    │                    │   ├── /mcp/data.conversation/:id
    │                    │   ├── /mcp/data.memories_list
    │                    │   └── /mcp/data.memory_file/:filename
    │                    │
    │                    └── Health & Status
    │                        └── /health
    │
    └── Components
        ├── App.tsx (Main app with view switcher)
        ├── DataExplorer.tsx (Data management UI)
        └── CodeEditor.tsx (Monaco editor wrapper)
```

## Usage Guide

### Uploading Data

1. Click **"Data Explorer"** button
2. Go to **"Status & Upload"** tab
3. Click **"Upload"** for conversations.json or memories.json
4. Select your exported JSON file
5. File is uploaded and overwrites any existing file

### Running Pipeline

1. Click **"Pipeline"** button
2. Scripts appear in order (1-6)
3. Click **"Run"** on any script
4. Output appears in the right panel
5. Status changes: Ready → Running → Complete/Failed

### Browsing Conversations

1. Go to **"Data Explorer"** → **"Conversations"** tab
2. Search bar filters by title or ID
3. Click any conversation to open in Monaco editor
4. Edit the JSONL content
5. Click **"Save"** to persist changes

### Browsing Memories

1. Go to **"Data Explorer"** → **"Memories"** tab
2. Click any memory file to open in Monaco editor
3. Edit and save changes

### Cleaning Data

1. Go to **"Data Explorer"** → **"Status & Upload"** tab
2. Click **"Clean"** next to any directory
3. Confirm the action
4. Generated files removed (source files preserved)

## Build for Production

```bash
npm run build
npm run preview
```

Output in `dist/` can be served by any static file server.

