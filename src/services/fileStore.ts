import fs from "fs";
import path from "path";
import { z } from "zod";
import { config } from "../config";
import { logger } from "../utils/logger";
import { MemoryFileName, MemoryRecord } from "../mcp/types";

const memoryRecordSchema = z.object({
  id: z.string(),
  type: z.string(),
}).passthrough();

function ensureMemoryDir() {
  if (!fs.existsSync(config.MEMORY_DIR)) {
    fs.mkdirSync(config.MEMORY_DIR, { recursive: true });
    logger.info("Created memory directory", config.MEMORY_DIR);
  }
}

function getFilePath(file: MemoryFileName): string {
  ensureMemoryDir();
  // Add .jsonl extension if not present
  const filename = file.endsWith('.jsonl') ? file : `${file}.jsonl`;
  return path.join(config.MEMORY_DIR, filename);
}

/**
 * Dynamically list all .jsonl files in the memory directory
 */
export function listMemoryFiles(): MemoryFileName[] {
  ensureMemoryDir();
  
  try {
    const files = fs.readdirSync(config.MEMORY_DIR);
    return files
      .filter(f => f.endsWith('.jsonl'))
      .map(f => f.replace('.jsonl', ''));
  } catch (err) {
    logger.warn("Error listing memory files", { error: String(err) });
    return [];
  }
}

export async function readAllRecords(file: MemoryFileName): Promise<MemoryRecord[]> {
  const filePath = getFilePath(file);

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
      logger.warn("Skipping invalid memory record line", { file, line: line.slice(0, 100), error: String(err) });
    }
  }

  return records;
}

export async function appendRecord(file: MemoryFileName, record: MemoryRecord): Promise<void> {
  const filePath = getFilePath(file);
  const validated = memoryRecordSchema.parse(record);
  const line = JSON.stringify(validated) + "\n";
  await fs.promises.appendFile(filePath, line, "utf8");
}

/**
 * Read all records from ALL memory files
 */
export async function readAllMemoryRecords(): Promise<{ file: string; records: MemoryRecord[] }[]> {
  const files = listMemoryFiles();
  const results: { file: string; records: MemoryRecord[] }[] = [];
  
  for (const file of files) {
    try {
      const records = await readAllRecords(file);
      results.push({ file, records });
    } catch (err) {
      logger.warn("Error reading memory file", { file, error: String(err) });
    }
  }
  
  return results;
}
