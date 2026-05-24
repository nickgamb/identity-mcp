import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";
import { config } from "../config";
import { logger } from "./logger";
import { PYTHON_CMD } from "./pythonCmd";

const execFileAsync = promisify(execFile);

const CSV_CORPUS_SCRIPT = path.join(
  config.PROJECT_ROOT,
  "scripts/conversation_processing/csv_corpus.py"
);

const TABULAR_EXTENSIONS = new Set([".csv", ".tsv"]);

/** 50MB stdout cap for large tabular exports */
const MAX_BUFFER = 50 * 1024 * 1024;

export function isTabularFile(filepath: string): boolean {
  return TABULAR_EXTENSIONS.has(path.extname(filepath).toLowerCase());
}

export async function loadTabularCorpusText(fullPath: string): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync(
      PYTHON_CMD,
      [CSV_CORPUS_SCRIPT, "corpus", fullPath],
      { maxBuffer: MAX_BUFFER, timeout: 120_000, encoding: "utf8" }
    );
    return stdout;
  } catch (error) {
    logger.warn("csv_corpus corpus failed, falling back to raw file", {
      fullPath,
      error: String(error),
    });
    return null;
  }
}

export async function searchTabularRows(
  fullPath: string,
  query: string,
  limit = 20
): Promise<string[] | null> {
  try {
    const { stdout } = await execFileAsync(
      PYTHON_CMD,
      [CSV_CORPUS_SCRIPT, "search", fullPath, query, String(limit)],
      { maxBuffer: MAX_BUFFER, timeout: 120_000, encoding: "utf8" }
    );
    const parsed = JSON.parse(stdout.trim());
    return Array.isArray(parsed) ? parsed.map(String) : [];
  } catch (error) {
    logger.warn("csv_corpus search failed", { fullPath, error: String(error) });
    return null;
  }
}
