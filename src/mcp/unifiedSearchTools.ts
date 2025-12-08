/**
 * Unified search across memories, files, and conversations
 */

import { handleMemorySearch, MemorySearchRequest } from "./memorySearchTools";
import { handleFileSearch, FileSearchRequest } from "./fileTools";
import { handleConversationSearch, ConversationSearchRequest } from "./conversationTools";
import { logger } from "../utils/logger";

export interface UnifiedSearchRequest {
  query: string;
  sources?: Array<"memories" | "files" | "conversations">;
  limit?: number; // Per source
}

export interface UnifiedSearchResponse {
  memories: {
    results: Array<{
      file: string;
      record: any;
      relevance: number;
    }>;
    count: number;
  };
  files: {
    results: Array<{
      filepath: string;
      content: string;
      relevance?: number;
    }>;
    count: number;
  };
  conversations: {
    conversations: Array<{
      conversationId: string;
      messages: Array<{
        role: string;
        content: string;
        timestamp: string;
      }>;
    }>;
    count: number;
  };
  totalResults: number;
}

export async function handleUnifiedSearch(
  req: UnifiedSearchRequest
): Promise<UnifiedSearchResponse> {
  try {
    const sources = req.sources || ["memories", "files", "conversations"];
    const limit = req.limit || 20;
    
    const results: UnifiedSearchResponse = {
      memories: { results: [], count: 0 },
      files: { results: [], count: 0 },
      conversations: { conversations: [], count: 0 },
      totalResults: 0,
    };
    
    // Search memories
    if (sources.includes("memories")) {
      try {
        const memoryResults = await handleMemorySearch({
          query: req.query,
          limit,
        });
        results.memories = {
          results: memoryResults.results.map((r) => ({
            file: r.file,
            record: r.record,
            relevance: r.relevance,
          })),
          count: memoryResults.count,
        };
      } catch (error) {
        logger.warn("Error in unified memory search", error);
      }
    }
    
    // Search files
    if (sources.includes("files")) {
      try {
        const fileResults = await handleFileSearch({
          query: req.query,
        });
        results.files = {
          results: (fileResults.results || []).slice(0, limit).map((r: any) => ({
            filepath: r.filepath,
            content: r.content || r.text || "",
            relevance: r.relevance,
          })),
          count: fileResults.results?.length || 0,
        };
      } catch (error) {
        logger.warn("Error in unified file search", error);
      }
    }
    
    // Search conversations
    if (sources.includes("conversations")) {
      try {
        const convResults = await handleConversationSearch({
          query: req.query,
          limit,
        });
        results.conversations = {
          conversations: convResults.conversations.map((c) => ({
            conversationId: c.conversationId,
            messages: c.messages.map((m) => ({
              role: m.role,
              content: m.content,
              timestamp: m.timestamp,
            })),
          })),
          count: convResults.count,
        };
      } catch (error) {
        logger.warn("Error in unified conversation search", error);
      }
    }
    
    results.totalResults = 
      results.memories.count + 
      results.files.count + 
      results.conversations.count;
    
    return results;
  } catch (error) {
    logger.error("Error in unified search", { query: req.query, error });
    return {
      memories: { results: [], count: 0 },
      files: { results: [], count: 0 },
      conversations: { conversations: [], count: 0 },
      totalResults: 0,
    };
  }
}

