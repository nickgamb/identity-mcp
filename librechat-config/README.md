# LibreChat Configuration

This directory contains the LibreChat configuration file that connects LibreChat to:
- **Ollama** server (for local LLM inference)
- **Identity MCP** server (for memory and conversation context)

## Configuration

The `librechat.yaml` file configures:
- Ollama as a custom endpoint provider
- Available models from your Ollama installation
- Connection settings for the Ollama API

## Accessing LibreChat

Once the services are running:
1. Open your browser to `http://localhost:3080`
2. Register a new account (registration is enabled by default)
3. Select "Ollama" as your endpoint
4. Choose a model from your available Ollama models

## Using Identity MCP

The MCP server is available at `http://localhost:4000` and provides:
- Identity retrieval (`/mcp/identity.get_core`, `/mcp/identity.get_full`)
- Memory operations (`/mcp/memory.list`, `/mcp/memory.get`, `/mcp/memory.append`)
- File RAG system (`/mcp/file.list`, `/mcp/file.get`, `/mcp/file.search`)

You can integrate MCP tools into LibreChat by:
1. Using custom functions/plugins
2. Making HTTP requests to the MCP endpoints
3. Using LibreChat's function calling capabilities (if supported)

## First Run

On first startup, LibreChat will:
1. Initialize the database
2. Create default admin user (check logs for credentials)
3. Set up search indexing

Check the logs if you encounter any issues:
```bash
docker-compose logs librechat-api
```

