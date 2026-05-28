/**
 * Conversation access tools for MCP
 * Provides read-only access to conversation history
 * Supports per-user conversation directories for multi-user mode
 */

import { ConversationLoader, ConversationSequence } from "../services/conversationLoader";
import { logger } from "../utils/logger";
import { tokenize } from "../utils/queryTokens";

export interface ConversationListRequest {
  limit?: number;
  offset?: number;
}

export interface ConversationListResponse {
  conversations: Array<{
    conversationId: string;
    messageCount: number;
    firstMessage?: string;
    lastMessage?: string;
    filePath?: string;
  }>;
  total: number;
}

export interface ConversationGetRequest {
  conversationId: string;
}

export interface ConversationGetResponse {
  conversation: ConversationSequence | null;
}

export interface ConversationSearchRequest {
  query: string;
  limit?: number;
}

export interface ConversationSearchResponse {
  conversations: ConversationSequence[];
  count: number;
}

export interface ConversationByDateRangeRequest {
  startDate?: string; // ISO date string
  endDate?: string; // ISO date string
  limit?: number;
}

export interface ConversationByDateRangeResponse {
  conversations: ConversationSequence[];
  count: number;
}

// Note: Loader is created per-request with user context for multi-user support
// For backward compatibility, userId can be null (single-user mode)

export async function handleConversationList(
  req: ConversationListRequest,
  userId: string | null = null
): Promise<ConversationListResponse> {
  try {
    const loader = new ConversationLoader(undefined, userId);
    const all = await loader.loadAllConversations();
    const total = all.length;
    
    const limit = req.limit ?? 100;
    const offset = req.offset ?? 0;
    const slice = all.slice(offset, offset + limit);
    
    const conversations = slice.map((conv) => ({
      conversationId: conv.conversationId,
      messageCount: conv.messages.length,
      firstMessage: conv.metadata?.firstMessage,
      lastMessage: conv.metadata?.lastMessage,
      filePath: conv.metadata?.filePath,
    }));
    
    return { conversations, total };
  } catch (error) {
    logger.error("Error listing conversations", error);
    return { conversations: [], total: 0 };
  }
}

export async function handleConversationGet(
  req: ConversationGetRequest,
  userId: string | null = null
): Promise<ConversationGetResponse> {
  try {
    const loader = new ConversationLoader(undefined, userId);
    const conversation = await loader.loadConversationById(req.conversationId);
    return { conversation };
  } catch (error) {
    logger.error("Error getting conversation", { conversationId: req.conversationId, error });
    return { conversation: null };
  }
}

export async function handleConversationSearch(
  req: ConversationSearchRequest,
  userId: string | null = null
): Promise<ConversationSearchResponse> {
  try {
    const matcher = tokenize(req.query);
    if (matcher.isEmpty) return { conversations: [], count: 0 };

    const loader = new ConversationLoader(undefined, userId);
    const all = await loader.loadAllConversations();
    const limit = req.limit ?? 50;

    // Score each conversation by total token hits across id + messages.
    // Hit per-message saturates at full token count so a single dense
    // message doesn't dominate over a conversation with broader coverage.
    const scored: Array<{ conv: ConversationSequence; score: number }> = [];

    for (const conv of all) {
      let score = matcher.matchCount(conv.conversationId);
      for (const msg of conv.messages) {
        score += matcher.matchCount(msg.content);
      }
      if (score > 0) scored.push({ conv, score });
    }

    scored.sort((a, b) => b.score - a.score);

    return {
      conversations: scored.slice(0, limit).map((s) => s.conv),
      count: scored.length,
    };
  } catch (error) {
    logger.error("Error searching conversations", { query: req.query, error });
    return { conversations: [], count: 0 };
  }
}

export async function handleConversationByDateRange(
  req: ConversationByDateRangeRequest,
  userId: string | null = null
): Promise<ConversationByDateRangeResponse> {
  try {
    const loader = new ConversationLoader(undefined, userId);
    const all = await loader.loadAllConversations();
    const limit = req.limit ?? 100;
    
    let filtered = all;
    
    if (req.startDate || req.endDate) {
      const startTime = req.startDate ? new Date(req.startDate).getTime() : 0;
      const endTime = req.endDate ? new Date(req.endDate).getTime() : Date.now();
      
      filtered = all.filter((conv) => {
        if (conv.messages.length === 0) return false;
        
        const firstMsg = conv.messages[0];
        const lastMsg = conv.messages[conv.messages.length - 1];
        
        const firstTime = firstMsg.timestamp ? new Date(firstMsg.timestamp).getTime() : 0;
        const lastTime = lastMsg.timestamp ? new Date(lastMsg.timestamp).getTime() : 0;
        
        // Include if conversation overlaps with date range
        return (firstTime >= startTime && firstTime <= endTime) ||
               (lastTime >= startTime && lastTime <= endTime) ||
               (firstTime <= startTime && lastTime >= endTime);
      });
    }
    
    const conversations = filtered.slice(0, limit);
    
    return { conversations, count: filtered.length };
  } catch (error) {
    logger.error("Error getting conversations by date range", { req, error });
    return { conversations: [], count: 0 };
  }
}

