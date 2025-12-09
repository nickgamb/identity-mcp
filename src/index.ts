import express, { Request, Response } from "express";
import { config } from "./config";
import { logger } from "./utils/logger";
import { healthRouter } from "./routes/health";
import { mcpRouter } from "./routes/httpApi";
import { mcpProtocolRouter } from "./routes/mcpProtocol";
import { MemoryParser } from "./services/memoryParser";

const app = express();

// CORS middleware for MCP protocol
app.use((req: Request, res: Response, next) => {
  // Allow CORS for MCP protocol endpoints
  if (req.path.startsWith("/mcp-protocol")) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    // Allow standard headers plus those used by HTTP MCP + SSE reconnect
    res.header(
      "Access-Control-Allow-Headers",
      [
        "Content-Type",
        "Accept",
        "Cache-Control",
        // MCP session + event headers
        "mcp-session-id",
        "MCP-Session-Id",
        "Last-Event-ID",
        "last-event-id",
        // Common auth / fetch headers in case LibreChat or other clients add them
        "Authorization",
        "X-Requested-With",
      ].join(", "),
    );
    res.header("Access-Control-Expose-Headers", "mcp-session-id");
    
    // For SSE streams, set headers to keep connection alive
    if (req.method === "GET") {
      res.header("Cache-Control", "no-cache");
      res.header("Connection", "keep-alive");
      res.header("X-Accel-Buffering", "no"); // Disable nginx buffering if present
    }
    
    if (req.method === "OPTIONS") {
      return res.sendStatus(200);
    }
  }
  next();
});

app.use(express.json({ limit: "500mb" }));

app.use(healthRouter);
app.use(mcpRouter);
// MCP Protocol endpoint for LibreChat compatibility
app.use("/mcp-protocol", mcpProtocolRouter);

app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: "not_found",
    path: req.path,
  });
});

// Auto-parse memories.json on startup if it exists and user.context.jsonl doesn't
async function initializeMemories() {
  try {
    const parser = new MemoryParser();
    const contextPath = require("path").join(config.MEMORY_DIR, "user.context.jsonl");
    const memoriesPath = require("path").join(config.MEMORY_DIR, "memories.json");
    const fs = require("fs");

    // Only parse if memories.json exists and user.context.jsonl doesn't exist or is empty
    if (fs.existsSync(memoriesPath)) {
      if (!fs.existsSync(contextPath) || fs.readFileSync(contextPath, "utf8").trim().length === 0) {
        logger.info("Auto-parsing memories.json on startup...");
        const count = await parser.parseChatGPTMemories();
        logger.info(`Auto-parsed ${count} memories from memories.json`);
      } else {
        logger.info("user.context.jsonl already exists, skipping auto-parse");
      }
    }
  } catch (error) {
    logger.warn("Failed to auto-parse memories.json on startup", error);
    // Don't fail startup if parsing fails
  }
}

app.listen(config.PORT, "0.0.0.0", async () => {
  logger.info(`Identity MCP listening on port ${config.PORT}`);
  // Initialize memories in background
  initializeMemories().catch(err => {
    logger.warn("Background memory initialization failed", err);
  });
});


