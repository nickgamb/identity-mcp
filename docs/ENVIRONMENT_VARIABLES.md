# Environment Variables

This document describes all environment variables used by the Identity MCP system, including OIDC/OAuth configuration.

## MCP Server Environment Variables

### Basic Configuration

- `PORT` (default: `4000`)
  - Port on which the MCP server listens

- `PROJECT_ROOT` (default: `./`)
  - Root directory of the project
  - Used for resolving relative paths

- `MEMORY_DIR` (default: `./memory`)
  - Directory where memory/identity data is stored
  - In multi-user mode, per-user subdirectories are created here

- `FILES_DIR` (default: `./files`)
  - Directory where uploaded files are stored
  - In multi-user mode, per-user subdirectories are created here

- `IDENTITY_SERVICE_URL` (default: `http://localhost:4001`)
  - URL of the Python identity verification service
  - Used for semantic identity verification

### OIDC Configuration

- `OIDC_ENABLED` (default: `false`)
  - Enable OIDC authentication
  - Set to `true` to enable multi-user support with OIDC

- `OIDC_ISSUER` (default: `http://localhost:8080/realms/mcp`)
  - OIDC issuer URL (Keycloak realm URL)
  - Should point to your Keycloak instance

- `OIDC_AUDIENCE` (default: `mcp-server`)
  - Expected audience claim in JWT tokens
  - Must match the client ID configured in Keycloak/Orchestrator

- `OIDC_REQUIRE_AUTH` (default: `false`)
  - Require authentication for all requests
  - Set to `true` to enforce authentication (reject anonymous requests)
  - Set to `false` for backward compatibility (allow anonymous access)

## Dashboard Environment Variables

These are Vite environment variables (prefixed with `VITE_`):

- `VITE_OIDC_ENABLED` (default: `false`)
  - Enable OIDC authentication in the dashboard
  - Set to `true` to show login/logout UI

- `VITE_OIDC_ISSUER` (default: `http://localhost:8081`)
  - OIDC issuer URL for the dashboard
  - Should point to Strata Orchestrator (OIDC provider)

- `VITE_OIDC_CLIENT_ID` (default: `dashboard`)
  - OIDC client ID for the dashboard
  - Must match the client configuration in Strata Orchestrator

- `VITE_OIDC_REDIRECT_URI` (default: `http://localhost:3001`)
  - OAuth redirect URI after login
  - Should match the dashboard URL

## Docker Compose Environment Variables

### Keycloak

- `KEYCLOAK_ADMIN` (default: `admin`)
  - Keycloak admin username

- `KEYCLOAK_ADMIN_PASSWORD` (default: `admin`)
  - Keycloak admin password
  - **Change this in production!**

### Strata Orchestrator

- `OIDC_ISSUER` (default: `http://keycloak:8080/realms/mcp`)
  - OIDC issuer URL (Keycloak) that Orchestrator connects to as a client

- `OIDC_CLIENT_ID` (default: `strata-orchestrator`)
  - Client ID for Orchestrator in Keycloak

- `OIDC_CLIENT_SECRET` (default: `change-me-in-production`)
  - Client secret for Orchestrator in Keycloak
  - **Change this in production!**

- `OIDC_PROVIDER_ISSUER` (default: `http://localhost:8081`)
  - OIDC issuer URL that Orchestrator provides to apps

- `OIDC_PROVIDER_AUDIENCE` (default: `mcp-server,librechat,dashboard`)
  - Comma-separated list of allowed audiences for tokens issued by Orchestrator

- `BACKEND_URL` (default: `http://mcp-server:4000`)
  - URL of the Identity MCP backend

- `OPA_POLICY_PATH` (default: `/app/policies`)
  - Path to OPA Rego policy files

### LibreChat

LibreChat has its own OIDC configuration. Check LibreChat documentation for exact variable names. Common variables:

- `LIBRECHAT_OIDC_ENABLED` (default: `false`)
  - Enable OIDC in LibreChat

- `LIBRECHAT_OIDC_ISSUER` (default: `http://localhost:8081`)
  - OIDC issuer URL (should point to Strata Orchestrator)

- `LIBRECHAT_OIDC_CLIENT_ID` (default: `librechat`)
  - OIDC client ID for LibreChat

- `LIBRECHAT_OIDC_CLIENT_SECRET` (default: `change-me`)
  - OIDC client secret for LibreChat
  - **Change this in production!**

## Example Configuration

### Single-User Mode (Backward Compatible)

No OIDC variables needed. System works as before:

```bash
PORT=4000
MEMORY_DIR=./memory
FILES_DIR=./files
```

### Multi-User Mode with OIDC

```bash
# MCP Server
OIDC_ENABLED=true
OIDC_ISSUER=http://keycloak:8080/realms/mcp
OIDC_AUDIENCE=mcp-server
OIDC_REQUIRE_AUTH=false  # Allow anonymous for backward compat

# Dashboard
VITE_OIDC_ENABLED=true
VITE_OIDC_ISSUER=http://localhost:8081
VITE_OIDC_CLIENT_ID=dashboard
VITE_OIDC_REDIRECT_URI=http://localhost:3001
```

## Migration Path

1. **Start with single-user mode** (default): No OIDC variables needed
2. **Enable OIDC but allow anonymous**: Set `OIDC_ENABLED=true` but `OIDC_REQUIRE_AUTH=false`
3. **Enforce authentication**: Set `OIDC_REQUIRE_AUTH=true` when ready

This allows gradual migration without breaking existing deployments.

## Related Documentation

- **[Getting Started](./GETTING_STARTED.md)** - End-to-end setup with your ChatGPT data
- **[MCP Protocol Reference](./MCP_README.md)** - Complete API reference for all 50 MCP tools
- **[Identity Verification](./IDENTITY_VERIFICATION.md)** - How the verification system works
- **[Multi-User & OIDC Support](./MULTI_USER_OIDC.md)** - Multi-user data isolation and OIDC authentication
- **[Docker Setup](./DOCKER_SETUP.md)** - Container deployment guide
- **[Blog: Securing Identity MCP](./BLOG_SECURING_IDENTITY_MCP.md)** - Tutorial on adding OAuth/OIDC and policy-based access control
