import { MemoryParser } from "../services/memoryParser";

export interface MemoryParseRequest {
  force?: boolean; // Force re-parse even if user.context.jsonl exists
}

export interface MemoryParseResponse {
  ok: boolean;
  converted: number;
  message: string;
}

// Note: Parser is created per-request with user context for multi-user support

export async function handleMemoryParse(
  req: MemoryParseRequest,
  userId: string | null = null
): Promise<MemoryParseResponse> {
  try {
    const parser = new MemoryParser(userId);
    const converted = await parser.parseChatGPTMemories();
    
    return {
      ok: true,
      converted,
      message: `Successfully parsed ${converted} memories from memories.json to user.context.jsonl`,
    };
  } catch (error) {
    return {
      ok: false,
      converted: 0,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}

