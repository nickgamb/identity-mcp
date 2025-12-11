# Multi-User Support & OIDC Integration

The Identity MCP supports multi-user operation with complete data isolation through OIDC (OpenID Connect) authentication.

## Overview

When OIDC is enabled, each authenticated user's data is stored in isolated directories:
- `memory/{userId}/` - Memory records and analysis
- `conversations/{userId}/` - Conversation files
- `files/{userId}/` - RAG storage files
- `models/identity/{userId}/` - Trained identity models
- `training_data/{userId}/` - Fine-tuning datasets
- `adapters/{userId}/` - LoRA adapters

## Architecture

### Authentication Flow

```
User → Application (LibreChat/Dashboard) → OIDC Provider → Back to Application
```

When using an identity orchestrator (optional):
- Receives OIDC tokens from identity provider
- Downscopes tokens for least-privilege agent operations
- Mints ephemeral "on behalf of" tokens for MCP tool calls
- Enforces policy-based access control

### Data Isolation

All MCP tools, HTTP API routes, and Python scripts automatically use per-user paths when `userId` is present:

- **MCP Protocol Route**: Extracts `userId` from session context
- **HTTP API Routes**: Extract `userId` from JWT token via `getUserContext()`
- **Python Scripts**: Read `USER_ID` environment variable (set by `pipelineTools.ts`)
- **Identity Service**: Accepts `user_id` via query param, header, or request body

## Configuration

### Environment Variables

**MCP Server** (`src/config.ts`):
```bash
OIDC_ENABLED=true
OIDC_ISSUER=http://your-oidc-provider:8080/realms/your-realm
OIDC_CLIENT_ID=mcp-server
OIDC_CLIENT_SECRET=YOUR_MCP_CLIENT_SECRET
OIDC_REDIRECT_URI=http://localhost:4000/auth/callback
```

**Dashboard** (`.env` or `vite.config.ts`):
```bash
VITE_OIDC_ENABLED=true
VITE_OIDC_ISSUER=http://your-oidc-provider:8080/realms/your-realm
VITE_OIDC_CLIENT_ID=dashboard
VITE_OIDC_REDIRECT_URI=http://localhost:3001
```

### Docker Compose

See `docker-compose.security.yml` for example OIDC provider setup (customize for your chosen provider).

## Backward Compatibility

When OIDC is **disabled** or no user is authenticated:
- `userId` is `null`
- All paths use original single-user directories
- System behaves exactly as before

This ensures existing single-user deployments continue working without changes.

## Implementation Details

### User Context Extraction

The MCP server extracts user context from:
1. **MCP Protocol**: Stored per-session in `mcpProtocol.ts`
2. **HTTP API**: Extracted from JWT via `authMiddleware` → `getUserContext()`
3. **Python Scripts**: Passed via `USER_ID` environment variable

### Path Resolution

All services use `getUserDataPath()` utility:
```typescript
// Single-user: returns baseDir
// Multi-user: returns baseDir / userId
const userDir = getUserDataPath(config.MEMORY_DIR, userId);
```

### Identity Service

The Python identity service (`identity_service.py`):
- Caches models per user in memory
- Loads models on-demand when first accessed
- Accepts `user_id` via query param, `X-User-Id` header, or request body
- Falls back to default model if no `user_id` provided

### Python Scripts

All processing scripts automatically detect `USER_ID`:
```python
USER_ID = os.environ.get("USER_ID")
def get_user_dir(base_dir: Path, user_id: Optional[str] = None) -> Path:
    if user_id:
        return base_dir / user_id
    return base_dir

CONVERSATIONS_DIR = get_user_dir(PROJECT_ROOT / "conversations", USER_ID)
```

## Security Considerations

1. **Token Validation**: All tokens are validated via JWKS (JSON Web Key Set) from your OIDC provider
2. **Least Privilege**: Use an identity orchestrator to downscope tokens for agent operations (optional)
3. **Data Isolation**: File system paths ensure complete separation between users
4. **Optional Auth**: Middleware allows backward compatibility (anonymous access when OIDC disabled)

## Testing

To test multi-user support:

1. **Enable OIDC**: Set `OIDC_ENABLED=true` in environment
2. **Authenticate**: Login via dashboard or LibreChat
3. **Verify Isolation**: Check that data is stored in `{directory}/{userId}/` paths
4. **Test Scripts**: Run pipeline scripts and verify they use per-user paths

## Migration

Existing single-user deployments can migrate by:
1. Keeping `OIDC_ENABLED=false` (or unset) - continues working as-is
2. Enabling OIDC - new authenticated users get isolated directories
3. Optionally migrating existing data to a default user directory

No breaking changes required.

## Related Documentation

- **[Getting Started](./GETTING_STARTED.md)** - End-to-end setup with your ChatGPT data
- **[MCP Protocol Reference](./MCP_README.md)** - Complete API reference for all 50 MCP tools
- **[Identity Verification](./IDENTITY_VERIFICATION.md)** - How the verification system works
- **[Docker Setup](./DOCKER_SETUP.md)** - Container deployment guide
- **[Environment Variables](./ENVIRONMENT_VARIABLES.md)** - Complete reference for all configuration options
- **[Blog: Securing Identity MCP](./BLOG_SECURING_IDENTITY_MCP.md)** - Tutorial on adding OAuth/OIDC and policy-based access control
