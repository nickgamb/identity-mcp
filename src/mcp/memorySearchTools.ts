/**
 * Full-text memory search tools
 */

import { listMemoryFiles, readAllRecords, MemoryFileName } from "../services/fileStore";
import { MemoryRecord } from "./types";
import { logger } from "../utils/logger";
import { tokenize } from "../utils/queryTokens";

export interface MemorySearchRequest {
  query: string;
  files?: MemoryFileName[];
  limit?: number;
}

export interface MemorySearchResponse {
  results: Array<{
    file: MemoryFileName;
    record: MemoryRecord;
    relevance: number; // Simple relevance score (0-1)
  }>;
  count: number;
}

export async function handleMemorySearch(
  req: MemorySearchRequest,
  userId: string | null = null
): Promise<MemorySearchResponse> {
  try {
    const matcher = tokenize(req.query);
    if (matcher.isEmpty) return { results: [], count: 0 };

    const targetFiles: MemoryFileName[] = req.files && req.files.length > 0
      ? req.files
      : listMemoryFiles(userId);
    const limit = req.limit ?? 50;

    const results: Array<{
      file: MemoryFileName;
      record: MemoryRecord;
      relevance: number;
    }> = [];

    for (const file of targetFiles) {
      try {
        const records = await readAllRecords(file, userId);
        for (const record of records) {
          // Score = distinct tokens matched in the record's full JSON blob.
          // Normalized so a record matching every token scores 1.0.
          const blob = JSON.stringify(record);
          const hits = matcher.matchCount(blob);
          if (hits > 0) {
            results.push({
              file,
              record,
              relevance: hits / matcher.tokens.length,
            });
          }
        }
      } catch (error) {
        logger.warn("Error searching memory file", { file, error });
      }
    }

    results.sort((a, b) => b.relevance - a.relevance);

    return {
      results: results.slice(0, limit),
      count: results.length,
    };
  } catch (error) {
    logger.error("Error in memory search", { query: req.query, error });
    return { results: [], count: 0 };
  }
}

