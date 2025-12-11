/**
 * Statistics and analytics tools
 */

import { listMemoryFiles, readAllRecords, MemoryFileName } from "../services/fileStore";
import { ConversationLoader } from "../services/conversationLoader";
import { logger } from "../utils/logger";

export interface MemoryStatsRequest {
  files?: MemoryFileName[];
}

export interface MemoryStatsResponse {
  totalRecords: number;
  recordsByFile: Array<{
    file: MemoryFileName;
    count: number;
  }>;
  recordsByType: Record<string, number>;
  recordsByTag: Record<string, number>;
  dateRange?: {
    earliest?: string;
    latest?: string;
  };
}

export interface ConversationStatsRequest {
  // No parameters needed
}

export interface ConversationStatsResponse {
  totalConversations: number;
  totalMessages: number;
  averageMessagesPerConversation: number;
  dateRange?: {
    earliest?: string;
    latest?: string;
  };
  conversationsByYear: Record<string, number>;
}

export async function handleMemoryStats(
  req: MemoryStatsRequest,
  userId: string | null = null
): Promise<MemoryStatsResponse> {
  try {
    const targetFiles: MemoryFileName[] = req.files && req.files.length > 0 
      ? req.files 
      : listMemoryFiles(userId);
    
    let totalRecords = 0;
    const recordsByFile: Array<{ file: MemoryFileName; count: number }> = [];
    const recordsByType: Record<string, number> = {};
    const recordsByTag: Record<string, number> = {};
    const dates: number[] = [];
    
    for (const file of targetFiles) {
      try {
        const records = await readAllRecords(file, userId);
        const count = records.length;
        totalRecords += count;
        recordsByFile.push({ file, count });
        
        for (const record of records) {
          // Count by type
          const type = record.type || "unknown";
          recordsByType[type] = (recordsByType[type] || 0) + 1;
          
          // Count by tags
          const tags = (record as any).tags;
          if (Array.isArray(tags)) {
            for (const tag of tags) {
              recordsByTag[tag] = (recordsByTag[tag] || 0) + 1;
            }
          }
          
          // Collect dates
          const recordDate = (record as any).createdAt || (record as any).timestamp || (record as any).date;
          if (recordDate) {
            const dateTime = new Date(recordDate).getTime();
            if (!isNaN(dateTime)) {
              dates.push(dateTime);
            }
          }
        }
      } catch (error) {
        logger.warn("Error getting stats for memory file", { file, error });
      }
    }
    
    // Calculate date range
    let dateRange: { earliest?: string; latest?: string } | undefined;
    if (dates.length > 0) {
      dates.sort((a, b) => a - b);
      dateRange = {
        earliest: new Date(dates[0]).toISOString(),
        latest: new Date(dates[dates.length - 1]).toISOString(),
      };
    }
    
    return {
      totalRecords,
      recordsByFile,
      recordsByType,
      recordsByTag,
      dateRange,
    };
  } catch (error) {
    logger.error("Error getting memory stats", error);
    return {
      totalRecords: 0,
      recordsByFile: [],
      recordsByType: {},
      recordsByTag: {},
    };
  }
}

export async function handleConversationStats(
  req: ConversationStatsRequest,
  userId: string | null = null
): Promise<ConversationStatsResponse> {
  try {
    const loader = new ConversationLoader(undefined, userId);
    const conversations = await loader.loadAllConversations();
    
    let totalMessages = 0;
    const dates: number[] = [];
    const conversationsByYear: Record<string, number> = {};
    
    for (const conv of conversations) {
      totalMessages += conv.messages.length;
      
      if (conv.messages.length > 0) {
        const firstMsg = conv.messages[0];
        if (firstMsg.timestamp) {
          const dateTime = new Date(firstMsg.timestamp).getTime();
          if (!isNaN(dateTime)) {
            dates.push(dateTime);
            
            const year = new Date(firstMsg.timestamp).getFullYear().toString();
            conversationsByYear[year] = (conversationsByYear[year] || 0) + 1;
          }
        }
      }
    }
    
    const averageMessagesPerConversation = conversations.length > 0
      ? totalMessages / conversations.length
      : 0;
    
    // Calculate date range
    let dateRange: { earliest?: string; latest?: string } | undefined;
    if (dates.length > 0) {
      dates.sort((a, b) => a - b);
      dateRange = {
        earliest: new Date(dates[0]).toISOString(),
        latest: new Date(dates[dates.length - 1]).toISOString(),
      };
    }
    
    return {
      totalConversations: conversations.length,
      totalMessages,
      averageMessagesPerConversation: Math.round(averageMessagesPerConversation * 100) / 100,
      dateRange,
      conversationsByYear,
    };
  } catch (error) {
    logger.error("Error getting conversation stats", error);
    return {
      totalConversations: 0,
      totalMessages: 0,
      averageMessagesPerConversation: 0,
      conversationsByYear: {},
    };
  }
}

