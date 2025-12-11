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
  OIDC_ENABLED,
  OIDC_ISSUER,
  OIDC_AUDIENCE,
  OIDC_REQUIRE_AUTH
});

export const config = {
  PORT,
  PROJECT_ROOT,
  MEMORY_DIR,
  FILES_DIR,
  IDENTITY_SERVICE_URL,
  OIDC_ENABLED,
  OIDC_ISSUER,
  OIDC_AUDIENCE,
  OIDC_REQUIRE_AUTH,
};
