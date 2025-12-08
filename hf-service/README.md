# HuggingFace Inference Service

A FastAPI service that provides OpenAI-compatible API for HuggingFace models with proper tool/function calling support.

## Features

- **Lazy Loading**: Models only load into GPU memory on first request (saves VRAM)
- **Auto-Unload**: Models automatically unload after inactivity (default: 5 minutes)
- **Opt-In**: Service doesn't start by default (uses Docker Compose profile)
- **Tool Support**: Properly formats and handles tool calls for GLM and other models

## Starting the Service

The service uses a Docker Compose profile so it doesn't consume GPU memory when not needed:

```bash
# Start just the HF service
docker-compose --profile hf up -d hf-service

# Or start all services including HF
docker-compose --profile hf up -d
```

## Configuration

### Environment Variables

- `HF_MODEL`: HuggingFace model ID (default: `THUDM/glm-4-5-air`)
- `HF_KEEP_ALIVE`: Model keep-alive timeout in seconds
  - `0` = Unload immediately after each request
  - `-1` = Never unload (keep loaded indefinitely)
  - `300` = Unload after 5 minutes of inactivity (default)

### Example: Change Keep-Alive

Edit `docker-compose.yml`:

```yaml
environment:
  - HF_KEEP_ALIVE=600  # 10 minutes
```

Then restart: `docker-compose --profile hf up -d hf-service`

## Model Loading Behavior

1. **First Request**: Model loads into GPU memory (takes ~30-60 seconds)
2. **Subsequent Requests**: Model stays loaded (fast responses)
3. **After Inactivity**: Model unloads automatically (frees VRAM)
4. **Next Request**: Model reloads (takes ~30-60 seconds again)

## VRAM Usage

- GLM-4.5-Air: ~8-10GB VRAM (FP16)
- Model unloads when idle, so you can use Ollama models without conflict
- Both services can run simultaneously if you switch between them (only one model loaded at a time)

## Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "model": "THUDM/glm-4-5-air",
  "device": "cuda",
  "model_loaded": false,
  "keep_alive_seconds": 300
}
```

## API Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check (doesn't load model)

## Usage in LibreChat

The service is configured in `librechat-config/librechat.yaml` as the `HFService` endpoint. Select the "GLM-4.5-Air (With Tools)" preset to use it.

**Note**: Make sure the HF service is running before selecting this model in LibreChat:

```bash
docker-compose --profile hf up -d hf-service
```

