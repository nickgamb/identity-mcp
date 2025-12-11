import fs from "fs";
import path from "path";
import { z } from "zod";
import { config } from "../config";
import { logger } from "../utils/logger";
import { MemoryFileName, MemoryRecord } from "../mcp/types";
import { getUserDataPath, ensureUserDirectory } from "../utils/userContext";

const memoryRecordSchema = z.object({
  id: z.string(),
  type: z.string(),
}).passthrough();

/**
 * Get memory directory for a specific user (or base directory if anonymous)
 */
function getMemoryDir(userId: string | null = null): string {
  const baseDir = getUserDataPath(config.MEMORY_DIR, userId);
  ensureUserDirectory(baseDir);
  return baseDir;
}

function ensureMemoryDir(userId: string | null = null) {
  const memoryDir = getMemoryDir(userId);
  if (!fs.existsSync(memoryDir)) {
    fs.mkdirSync(memoryDir, { recursive: true });
    logger.info("Created memory directory", { memoryDir, userId });
  }
}

function getFilePath(file: MemoryFileName, userId: string | null = null): string {
  ensureMemoryDir(userId);
  // Add .jsonl extension if not present
  const filename = file.endsWith('.jsonl') ? file : `${file}.jsonl`;
  return path.join(getMemoryDir(userId), filename);
}

/**
 * Dynamically list all .jsonl files in the memory directory
 * @param userId - Optional user ID for multi-user support
 */
export function listMemoryFiles(userId: string | null = null): MemoryFileName[] {
  const memoryDir = getMemoryDir(userId);
  
  try {
    if (!fs.existsSync(memoryDir)) {
      return [];
    }
    const files = fs.readdirSync(memoryDir);
    return files
      .filter(f => f.endsWith('.jsonl'))
      .map(f => f.replace('.jsonl', ''));
  } catch (err) {
    logger.warn("Error listing memory files", { error: String(err), userId });
    return [];
  }
}

export async function readAllRecords(file: MemoryFileName, userId: string | null = null): Promise<MemoryRecord[]> {
  const filePath = getFilePath(file, userId);

  if (!fs.existsSync(filePath)) {
    // Auto-create empty file
    await fs.promises.writeFile(filePath, "", "utf8");
    return [];
  }

  const content = await fs.promises.readFile(filePath, "utf8");
  if (!content.trim()) {
    return [];
  }

  const lines = content.split("\n").filter((line: string) => line.trim().length > 0);
  const records: MemoryRecord[] = [];

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      const validated = memoryRecordSchema.parse(parsed);
      records.push(validated as MemoryRecord);
    } catch (err) {
      logger.warn("Skipping invalid memory record line", { file, line: line.slice(0, 100), error: String(err), userId });
    }
  }

  return records;
}

export async function appendRecord(file: MemoryFileName, record: MemoryRecord, userId: string | null = null): Promise<void> {
  const filePath = getFilePath(file, userId);
  const validated = memoryRecordSchema.parse(record);
  const line = JSON.stringify(validated) + "\n";
  await fs.promises.appendFile(filePath, line, "utf8");
}

/**
 * Read all records from ALL memory files
 * @param userId - Optional user ID for multi-user support
 */
export async function readAllMemoryRecords(userId: string | null = null): Promise<{ file: string; records: MemoryRecord[] }[]> {
  const files = listMemoryFiles(userId);
  const results: { file: string; records: MemoryRecord[] }[] = [];
  
  for (const file of files) {
    try {
      const records = await readAllRecords(file, userId);
      results.push({ file, records });
    } catch (err) {
      logger.warn("Error reading memory file", { file, error: String(err), userId });
    }
  }
  
  return results;
}
