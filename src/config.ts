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

logger.info("Config loaded", { PORT, MEMORY_DIR, FILES_DIR, IDENTITY_SERVICE_URL });

export const config = {
  PORT,
  PROJECT_ROOT,
  MEMORY_DIR,
  FILES_DIR,
  IDENTITY_SERVICE_URL,
};
