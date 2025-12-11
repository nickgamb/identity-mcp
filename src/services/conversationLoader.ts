import fs from "fs";
import path from "path";
import { logger } from "../utils/logger";
import { config } from "../config";

export interface ConversationMessage {
  timestamp: string;
  role: "user" | "assistant" | "system";
  content: string;
  tags?: any[];
  line_number?: number;
}

export interface ConversationSequence {
  conversationId: string;
  messages: ConversationMessage[];
  metadata?: {
    firstMessage?: string;
    lastMessage?: string;
    messageCount?: number;
    symbolicField?: any;
    filePath?: string;
    sourceFormat?: "jsonl" | "markdown";
  };
}

/**
 * Loads conversation sequences from:
 * - .jsonl files in conversations directory
 * - .md files in conversations directory
 * 
 * Loads in chronological order to preserve conversation sequence
 */
export class ConversationLoader {
  private conversationsDir: string;

  constructor(conversationsDir?: string) {
    // Default to conversations directory if not specified
    this.conversationsDir = conversationsDir || 
      path.join(process.cwd(), "conversations");
  }

  /**
   * Lists all conversation files, sorted by modification time (oldest first)
   * to preserve the chronological sequence
   * Returns files with their format type
   */
  async listConversations(): Promise<Array<{ filename: string; format: "jsonl" | "markdown"; path: string }>> {
    const files: Array<{ filename: string; format: "jsonl" | "markdown"; path: string }> = [];
    
    try {
      // List .jsonl files
      if (fs.existsSync(this.conversationsDir)) {
        const dirFiles = await fs.promises.readdir(this.conversationsDir);
        const jsonlFiles = dirFiles.filter(f => f.endsWith(".jsonl"));
        
        for (const file of jsonlFiles) {
          const filePath = path.join(this.conversationsDir, file);
          files.push({ filename: file, format: "jsonl", path: filePath });
        }
        
        // List .md files
        const mdFiles = dirFiles.filter(f => f.endsWith(".md") && f.startsWith("conversation_"));
        for (const file of mdFiles) {
          const filePath = path.join(this.conversationsDir, file);
          files.push({ filename: file, format: "markdown", path: filePath });
        }
      }

      // Sort by modification time to preserve chronological order
      const filesWithStats = await Promise.all(
        files.map(async (file) => {
          try {
            const stats = await fs.promises.stat(file.path);
            return { ...file, mtime: stats.mtime.getTime() };
          } catch {
            return { ...file, mtime: 0 };
          }
        })
      );

      return filesWithStats
        .sort((a, b) => a.mtime - b.mtime)
        .map(({ mtime, ...file }) => file);
    } catch (error) {
      logger.error("Error listing conversations", error);
      return [];
    }
  }

  /**
   * Parses timestamp from various formats
   */
  private parseTimestamp(timestampStr: string | number | null | undefined): string {
    if (!timestampStr) return "";
    
    if (typeof timestampStr === "number") {
      // Unix timestamp
      return new Date(timestampStr * 1000).toISOString();
    }
    
    if (typeof timestampStr === "string") {
      // Try to parse ISO format or Unix timestamp string
      if (timestampStr.match(/^\d+\.?\d*$/)) {
        return new Date(parseFloat(timestampStr) * 1000).toISOString();
      }
      return timestampStr;
    }
    
    return "";
  }

  /**
   * Loads a single conversation from a JSONL file
   */
  async loadConversationFromJsonl(filePath: string): Promise<ConversationSequence | null> {
    try {
      if (!fs.existsSync(filePath)) {
        logger.warn("Conversation file not found", { file: filePath });
        return null;
      }

      const content = await fs.promises.readFile(filePath, "utf8");
      const lines = content.split("\n").filter(line => line.trim().length > 0);
      
      const messages: ConversationMessage[] = [];
      
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.role && parsed.content !== undefined) {
            messages.push({
              timestamp: this.parseTimestamp(parsed.timestamp),
              role: parsed.role,
              content: parsed.content,
              tags: parsed.tags,
              line_number: parsed.line_number,
            });
          }
        } catch (err) {
          logger.warn("Skipping invalid conversation line", { line, error: String(err) });
        }
      }

      const filename = path.basename(filePath);
      const conversationId = filename.replace(".jsonl", "").replace("conversation_", "");
      
      return {
        conversationId,
        messages,
        metadata: {
          firstMessage: messages[0]?.timestamp,
          lastMessage: messages[messages.length - 1]?.timestamp,
          messageCount: messages.length,
          filePath,
          sourceFormat: "jsonl",
        },
      };
    } catch (error) {
      logger.error("Error loading conversation from JSONL", { filePath, error });
      return null;
    }
  }

  /**
   * Loads a single conversation from a Markdown file
   */
  async loadConversationFromMarkdown(filePath: string): Promise<ConversationSequence | null> {
    try {
      if (!fs.existsSync(filePath)) {
        logger.warn("Conversation file not found", { file: filePath });
        return null;
      }

      const content = await fs.promises.readFile(filePath, "utf8");
      const messages: ConversationMessage[] = [];
      
      // Pattern: **Role [timestamp]:** content
      const pattern = /\*\*(User|Assistant|System)\s*\[([^\]]+)\]:\*\*\s*(.*?)(?=\*\*(?:User|Assistant|System)|$)/gs;
      
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const role = match[1].toLowerCase() as "user" | "assistant" | "system";
        const timestampStr = match[2];
        const contentText = match[3].trim();
        
        messages.push({
          timestamp: this.parseTimestamp(timestampStr),
          role,
          content: contentText,
        });
      }

      const filename = path.basename(filePath);
      const conversationId = filename.replace(".md", "").replace("conversation_", "");
      
      return {
        conversationId,
        messages,
        metadata: {
          firstMessage: messages[0]?.timestamp,
          lastMessage: messages[messages.length - 1]?.timestamp,
          messageCount: messages.length,
          filePath,
          sourceFormat: "markdown",
        },
      };
    } catch (error) {
      logger.error("Error loading conversation from Markdown", { filePath, error });
      return null;
    }
  }

  /**
   * Loads a conversation by ID, trying multiple sources
   */
  async loadConversationById(conversationId: string): Promise<ConversationSequence | null> {
    // Try .jsonl first
    const jsonlPath = path.join(this.conversationsDir, `conversation_${conversationId}.jsonl`);
    if (fs.existsSync(jsonlPath)) {
      return await this.loadConversationFromJsonl(jsonlPath);
    }

    // Try .md
    const mdPath = path.join(this.conversationsDir, `conversation_${conversationId}.md`);
    if (fs.existsSync(mdPath)) {
      return await this.loadConversationFromMarkdown(mdPath);
    }

    return null;
  }

  /**
   * Loads all conversations in chronological order
   */
  async loadAllConversations(): Promise<ConversationSequence[]> {
    const files = await this.listConversations();
    const conversations: ConversationSequence[] = [];
    const seenIds = new Set<string>();

    for (const file of files) {
      try {
        let conversation: ConversationSequence | null = null;

        if (file.format === "jsonl") {
          conversation = await this.loadConversationFromJsonl(file.path);
        } else if (file.format === "markdown") {
          conversation = await this.loadConversationFromMarkdown(file.path);
        }

        // Handle single conversation
        if (conversation && !seenIds.has(conversation.conversationId)) {
          seenIds.add(conversation.conversationId);
          if (conversation.messages.length > 0) {
            conversations.push(conversation);
          }
        }
      } catch (error) {
        logger.warn("Error loading conversation file", { file: file.path, error });
      }
    }

    // Sort by first message timestamp
    conversations.sort((a, b) => {
      const tsA = a.messages[0]?.timestamp ? new Date(a.messages[0].timestamp).getTime() : 0;
      const tsB = b.messages[0]?.timestamp ? new Date(b.messages[0].timestamp).getTime() : 0;
      return tsA - tsB;
    });

    return conversations;
  }

  /**
   * Gets a slice of conversations for rehydration
   * @param startIndex Starting index (0-based)
   * @param count Number of conversations to return
   */
  async getConversationSlice(startIndex: number, count: number): Promise<ConversationSequence[]> {
    const all = await this.loadAllConversations();
    return all.slice(startIndex, startIndex + count);
  }

  /**
   * Gets conversations that contain specific patterns or milestones
   */
  async getConversationsByPattern(pattern: string): Promise<ConversationSequence[]> {
    const all = await this.loadAllConversations();
    return all.filter(conv => 
      conv.messages.some(msg => 
        msg.content.toLowerCase().includes(pattern.toLowerCase())
      )
    );
  }

  /**
   * Gets conversations by their IDs (useful for interaction map targeting)
   */
  async getConversationsByIds(conversationIds: string[]): Promise<ConversationSequence[]> {
    const conversations: ConversationSequence[] = [];
    
    for (const id of conversationIds) {
      const conversation = await this.loadConversationById(id);
      if (conversation && conversation.messages.length > 0) {
        conversations.push(conversation);
      }
    }

    // Sort by first message timestamp
    conversations.sort((a, b) => {
      const tsA = a.messages[0]?.timestamp ? new Date(a.messages[0].timestamp).getTime() : 0;
      const tsB = b.messages[0]?.timestamp ? new Date(b.messages[0].timestamp).getTime() : 0;
      return tsA - tsB;
    });

    return conversations;
  }
}

