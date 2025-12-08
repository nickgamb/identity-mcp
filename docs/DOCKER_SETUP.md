# Docker Setup Guide

Complete guide for running the Conversation Memory MCP with Ollama and LibreChat using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB of available RAM (16GB recommended)
- NVIDIA GPU with drivers (optional, for GPU acceleration)

## Quick Start

```bash
# Start all services
docker-compose up --build

# Services will be available at:
# - MCP Server: http://localhost:4000
# - Ollama: http://localhost:11434
# - LibreChat: http://localhost:3080
```

## Services

### MCP Server
- **Port**: 4000
- **Health**: `GET http://localhost:4000/health`
- **MCP Protocol**: `POST http://localhost:4000/mcp-protocol`
- **HTTP API**: `POST http://localhost:4000/mcp/*`

### Ollama Server
- **Port**: 11434
- **API**: `http://localhost:11434`
- **Models**: Auto-discovered from `ollama_models/`

### LibreChat
- **Port**: 3080
- **Web UI**: `http://localhost:3080`
- **Database**: MongoDB (internal)
- **Cache**: Redis (internal)
- **Search**: Meilisearch (internal)

## First-Time Setup

### 1. Access LibreChat

1. Navigate to `http://localhost:3080`
2. Register a new account (registration enabled by default)
3. Select "LocalOllama" as your endpoint
4. Choose a model from available Ollama models

### 2. Configure MCP in LibreChat

The MCP server is configured in `librechat-config/librechat.yaml`:

```yaml
mcpServers:
  memory-mcp:
    type: http
    url: http://mcp-server:4000/mcp-protocol
    timeout: 120000
```

### 3. Verify Services

```bash
# Check MCP Server
curl http://localhost:4000/health

# Check Ollama
curl http://localhost:11434/api/tags

# Check LibreChat (should return HTML)
curl http://localhost:3080
```

## Volumes

Data is persisted in Docker volumes:
- `ollama-data` - Ollama models and data
- `mongodb-data` - LibreChat database
- `redis-data` - Redis cache
- `meilisearch-data` - Search index

Local directories are mounted:
- `./memory` → `/app/memory` (memory files)
- `./conversations` → `/app/conversations` (conversation history)
- `./files` → `/app/files` (RAG-able files)
- `./ollama_models` → `/root/.ollama/models/blobs` (Ollama models)

## Troubleshooting

### LibreChat won't start
```bash
# Check logs
docker-compose logs librechat-api

# Ensure dependencies are healthy
docker-compose ps
```

### Ollama models not showing
```bash
# Check Ollama logs
docker-compose logs ollama-server

# Verify models directory
docker exec ollama-server ls /root/.ollama/models/blobs

# Pull a model manually
docker exec ollama-server ollama pull llama3:8b
```

### MCP not responding
```bash
# Check health
curl http://localhost:4000/health

# View logs
docker-compose logs mcp-server

# Verify volumes mounted
docker exec mcp-server ls /app/memory
```

### Port conflicts

Modify `docker-compose.yml` to change ports:
```yaml
ports:
  - "4001:4000"  # Change external port
```

## Stopping Services

```bash
# Stop services (keeps data)
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v
```

## Updating Services

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up --build -d
```

## Network

All services are on the `mcp-network` bridge network:
- Services communicate using service names (e.g., `http://ollama:11434`)
- External access via published ports
- Internal communication is isolated

## GPU Support

If you have an NVIDIA GPU, Ollama will automatically use it. The `docker-compose.yml` includes:

```yaml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

Ensure NVIDIA Container Toolkit is installed on your host.

## Next Steps

1. **Test MCP endpoints** using curl or Postman
2. **Search your conversations** with the conversation tools
3. **Query your memories** with memory_search
4. **Try unified search** with search_all

See [README.md](./README.md) for full API documentation.
