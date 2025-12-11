/**
 * Full-text memory search tools
 */

import { listMemoryFiles, readAllRecords, MemoryFileName } from "../services/fileStore";
import { MemoryRecord } from "./types";
import { logger } from "../utils/logger";

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
    const query = req.query.toLowerCase();
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
          let relevance = 0;
          let matchCount = 0;
          
          // Search in all string fields
          const searchableFields = [
            record.id,
            record.type,
            (record as any).content,
            (record as any).text,
            (record as any).message,
            (record as any).title,
            (record as any).description,
            JSON.stringify(record), // Fallback: search entire record
          ].filter(Boolean).map(String);
          
          for (const field of searchableFields) {
            const fieldLower = field.toLowerCase();
            if (fieldLower.includes(query)) {
              matchCount++;
              // Higher relevance for exact matches and matches in important fields
              if (fieldLower === query) {
                relevance += 1.0;
              } else if (fieldLower.startsWith(query)) {
                relevance += 0.8;
              } else {
                relevance += 0.5;
              }
            }
          }
          
          // Normalize relevance (0-1)
          if (matchCount > 0) {
            relevance = Math.min(1.0, relevance / matchCount);
            
            results.push({
              file,
              record,
              relevance,
            });
          }
        }
      } catch (error) {
        logger.warn("Error searching memory file", { file, error });
      }
    }
    
    // Sort by relevance (highest first)
    results.sort((a, b) => b.relevance - a.relevance);
    
    // Apply limit
    const limited = results.slice(0, limit);
    
    return {
      results: limited,
      count: results.length,
    };
  } catch (error) {
    logger.error("Error in memory search", { query: req.query, error });
    return { results: [], count: 0 };
  }
}

