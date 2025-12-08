import { Router, Request, Response } from "express";
import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";

const startedAt = Date.now();

export const healthRouter = Router();

healthRouter.get("/health", (_req: Request, res: Response) => {
  const uptimeSeconds = (Date.now() - startedAt) / 1000;
  const checks: Record<string, boolean | string> = {
    server: true,
    memoryDir: false,
    conversationsDir: false,
    filesDir: false,
  };

  // Check memory directory
  try {
    if (fs.existsSync(config.MEMORY_DIR)) {
      checks.memoryDir = true;
    }
  } catch (error) {
    logger.warn("Memory directory check failed", { error: String(error) });
  }

  // Check conversations directory
  try {
    const conversationsDir = path.join(process.cwd(), "conversations");
    if (fs.existsSync(conversationsDir)) {
      checks.conversationsDir = true;
    }
  } catch (error) {
    logger.warn("Conversations directory check failed", { error: String(error) });
  }

  // Check files directory
  try {
    const filesDir = path.join(process.cwd(), "files");
    if (fs.existsSync(filesDir)) {
      checks.filesDir = true;
    }
  } catch (error) {
    logger.warn("Files directory check failed", { error: String(error) });
  }

  const allHealthy = Object.values(checks).every(v => v === true);
  const status = allHealthy ? 200 : 503;

  res.status(status).json({
    status: allHealthy ? "ok" : "degraded",
    uptime: Number(uptimeSeconds.toFixed(2)),
    pid: process.pid,
    checks,
  });
});


