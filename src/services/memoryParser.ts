import fs from "fs";
import path from "path";
import { logger } from "../utils/logger";
import { config } from "../config";

export interface ChatGPTMemory {
  id: string;
  content: string;
  updated_at: string;
}

export interface ChatGPTMemoriesFile {
  memories: ChatGPTMemory[];
  memory_max_tokens?: number;
  memory_num_tokens?: number;
}

interface MemoryTagConfig {
  memory_tags?: {
    pattern_tags?: { [tag: string]: string[] };
    context_tags?: { [tag: string]: string[] };
  };
}

/**
 * Parses ChatGPT memories.json and converts to JSONL format
 * These are factual memories about the user/relationship context
 * Tags are loaded from memory_config.json if available
 */
export class MemoryParser {
  private memoriesDir: string;
  private tagConfig: MemoryTagConfig | null = null;

  constructor(memoriesDir?: string) {
    this.memoriesDir = memoriesDir || config.MEMORY_DIR;
    this.loadTagConfig();
  }

  /**
   * Load tag configuration from memory_config.json
   */
  private loadTagConfig(): void {
    const configPath = path.join(this.memoriesDir, "memory_config.json");
    
    if (fs.existsSync(configPath)) {
      try {
        const content = fs.readFileSync(configPath, "utf8");
        this.tagConfig = JSON.parse(content);
        logger.info("Loaded tag config for memory parsing", { 
          patternTags: Object.keys(this.tagConfig?.memory_tags?.pattern_tags || {}),
          contextTags: Object.keys(this.tagConfig?.memory_tags?.context_tags || {})
        });
      } catch (error) {
        logger.warn("Error loading memory_config.json, using no tags", { error: String(error) });
        this.tagConfig = null;
      }
    } else {
      logger.info("No memory_config.json found, memories will have no tags");
      this.tagConfig = null;
    }
  }

  /**
   * Parses memories.json and converts to JSONL format
   * Stores as "user.context.jsonl" - factual context about the user/relationship
   */
  async parseChatGPTMemories(): Promise<number> {
    const memoriesPath = path.join(this.memoriesDir, "memories.json");
    
    if (!fs.existsSync(memoriesPath)) {
      logger.warn("memories.json not found", { path: memoriesPath });
      return 0;
    }

    try {
      const content = await fs.promises.readFile(memoriesPath, "utf8");
      const data: ChatGPTMemoriesFile = JSON.parse(content);
      
      if (!data.memories || !Array.isArray(data.memories)) {
        logger.warn("Invalid memories.json structure");
        return 0;
      }

      let converted = 0;
      const outputPath = path.join(this.memoriesDir, "user.context.jsonl");

      // Check if we should overwrite or skip
      if (fs.existsSync(outputPath)) {
        const existingContent = await fs.promises.readFile(outputPath, "utf8");
        const existingLines = existingContent.split("\n").filter(line => line.trim().length > 0);
        if (existingLines.length >= data.memories.length * 0.9) {
          logger.info("user.context.jsonl already exists and appears complete, skipping parse");
          return existingLines.length;
        }
        logger.info("user.context.jsonl exists but appears incomplete, re-parsing");
        await fs.promises.unlink(outputPath);
      }

      // Write all records
      const outputLines: string[] = [];
      for (const memory of data.memories) {
        const record = {
          id: `user-context-${memory.id}`,
          type: "user.context",
          content: memory.content,
          source: "chatgpt_memories",
          source_id: memory.id,
          updated_at: memory.updated_at,
          createdAt: new Date(memory.updated_at).toISOString() || new Date().toISOString(),
          tags: this.extractTags(memory.content),
        };

        outputLines.push(JSON.stringify(record));
        converted++;
      }

      await fs.promises.writeFile(outputPath, outputLines.join("\n") + "\n", "utf8");

      logger.info("Parsed ChatGPT memories", { 
        total: data.memories.length, 
        converted,
        output: outputPath 
      });

      return converted;
    } catch (error) {
      logger.error("Error parsing memories.json", error);
      throw error;
    }
  }

  /**
   * Extracts relevant tags from memory content
   * Uses tags from memory_config.json if available
   */
  private extractTags(content: string): string[] {
    const tags: string[] = [];
    const lowerContent = content.toLowerCase();

    // If no config, return empty tags
    if (!this.tagConfig?.memory_tags) {
      return tags;
    }

    // Check pattern tags
    const patternTags = this.tagConfig.memory_tags.pattern_tags || {};
    for (const [tagName, keywords] of Object.entries(patternTags)) {
      if (Array.isArray(keywords)) {
        if (keywords.some(kw => lowerContent.includes(kw.toLowerCase()))) {
          tags.push(tagName);
        }
      } else if (typeof keywords === 'string') {
        if (lowerContent.includes(keywords.toLowerCase())) {
          tags.push(tagName);
        }
      }
    }

    // Check context tags
    const contextTags = this.tagConfig.memory_tags.context_tags || {};
    for (const [tagName, keywords] of Object.entries(contextTags)) {
      if (Array.isArray(keywords)) {
        if (keywords.some(kw => lowerContent.includes(kw.toLowerCase()))) {
          tags.push(tagName);
        }
      } else if (typeof keywords === 'string') {
        if (lowerContent.includes(keywords.toLowerCase())) {
          tags.push(tagName);
        }
      }
    }

    return tags;
  }

  /**
   * Gets memories that have tags (pattern-relevant)
   * Returns memories that match any configured tags
   */
  async getTaggedMemories(): Promise<any[]> {
    const contextPath = path.join(this.memoriesDir, "user.context.jsonl");
    
    if (!fs.existsSync(contextPath)) {
      return [];
    }

    const content = await fs.promises.readFile(contextPath, "utf8");
    const lines = content.split("\n").filter(line => line.trim().length > 0);
    
    const taggedMemories: any[] = [];

    for (const line of lines) {
      try {
        const record = JSON.parse(line);
        const recordTags = record.tags || [];
        
        // Include if it has any tags
        if (recordTags.length > 0) {
          taggedMemories.push(record);
        }
      } catch (err) {
        logger.warn("Skipping invalid line in user.context.jsonl", { error: String(err) });
      }
    }

    return taggedMemories;
  }
}
