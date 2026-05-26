import dotenv from "dotenv";
import path from "path";
import { logger } from "./utils/logger";

dotenv.config();

const PORT = parseInt(process.env.PORT || "4000", 10);
const PROJECT_ROOT = process.env.PROJECT_ROOT || path.resolve(__dirname, "..");
const MEMORY_DIR = process.env.MEMORY_DIR || path.join(PROJECT_ROOT, "memory");
const FILES_DIR = process.env.FILES_DIR || path.join(PROJECT_ROOT, "files");

// Identity verification service URL (Python semantic service)
const IDENTITY_SERVICE_URL = process.env.IDENTITY_SERVICE_URL || "http://localhost:4001";

// Letta Configuration (optional - for memory system)
const LETTA_BASE_URL = process.env.LETTA_BASE_URL || "http://letta:8283";
const LETTA_AGENT_NAME = process.env.LETTA_AGENT_NAME || "identity";
/** URL baked into Letta tool code — must be reachable from the letta container, not localhost. */
const MCP_SERVER_URL =
  process.env.MCP_SERVER_URL || "http://mcp-server:4000";
const OLLAMA_BASE_URL =
  process.env.OLLAMA_BASE_URL || "http://ollama:11434";

// Reverie (background self-reflection)
const REVERIE_ENABLED = process.env.REVERIE_ENABLED === "true";
const REVERIE_INTERVAL_MINUTES = parseInt(process.env.REVERIE_INTERVAL_MINUTES || "120", 10);
const REVERIE_MAX_STEPS = Math.max(3, parseInt(process.env.REVERIE_MAX_STEPS || "8", 10));
const REVERIE_TIMEOUT_MS = parseInt(process.env.REVERIE_TIMEOUT_MS || "600000", 10);

// OIDC Configuration (optional - for multi-user support)
const OIDC_ENABLED = process.env.OIDC_ENABLED === "true";
const OIDC_ISSUER = process.env.OIDC_ISSUER || "http://localhost:8080/realms/mcp";
const OIDC_AUDIENCE = process.env.OIDC_AUDIENCE || "mcp-server";
const OIDC_REQUIRE_AUTH = process.env.OIDC_REQUIRE_AUTH === "true"; // If false, allows anonymous access (backward compat)

logger.info("Config loaded", {
  PORT,
  MEMORY_DIR,
  FILES_DIR,
  IDENTITY_SERVICE_URL,
  LETTA_BASE_URL,
  LETTA_AGENT_NAME,
  OIDC_ENABLED,
  OIDC_ISSUER,
  OIDC_AUDIENCE,
  OIDC_REQUIRE_AUTH,
  REVERIE_ENABLED,
  REVERIE_INTERVAL_MINUTES,
  REVERIE_MAX_STEPS,
  REVERIE_TIMEOUT_MS,
});

export const config = {
  PORT,
  PROJECT_ROOT,
  MEMORY_DIR,
  FILES_DIR,
  IDENTITY_SERVICE_URL,
  LETTA_BASE_URL,
  LETTA_AGENT_NAME,
  MCP_SERVER_URL,
  OLLAMA_BASE_URL,
  OIDC_ENABLED,
  OIDC_ISSUER,
  OIDC_AUDIENCE,
  OIDC_REQUIRE_AUTH,
  REVERIE_ENABLED,
  REVERIE_INTERVAL_MINUTES,
  REVERIE_MAX_STEPS,
  REVERIE_TIMEOUT_MS,
};
