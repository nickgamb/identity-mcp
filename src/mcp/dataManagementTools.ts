/**
 * Data Management Tools
 * 
 * Tools for managing source files (conversations.json, memories.json),
 * browsing and editing generated data, and cleaning directories.
 */

import fs from "fs";
import path from "path";

const PROJECT_ROOT = process.cwd();

// ============================================================================
// Status & Upload
// ============================================================================

export interface DataStatusResponse {
  sourceFiles: {
    conversationsJson: boolean;
    memoriesJson: boolean;
  };
  generatedData: {
    conversations: boolean;
    memory: boolean;
    identityModel: boolean;
    interactionMap: boolean;
  };
  counts: {
    conversationFiles: number;
    memoryFiles: number;
    files: number;
  };
}

export async function handleDataStatus(): Promise<DataStatusResponse> {
  const status: DataStatusResponse = {
    sourceFiles: {
      conversationsJson: fs.existsSync(path.join(PROJECT_ROOT, "conversations", "conversations.json")),
      memoriesJson: fs.existsSync(path.join(PROJECT_ROOT, "memory", "memories.json")),
    },
    generatedData: {
      conversations: fs.existsSync(path.join(PROJECT_ROOT, "conversations")) && 
        fs.readdirSync(path.join(PROJECT_ROOT, "conversations")).filter((f: string) => f.endsWith(".jsonl")).length > 0,
      memory: fs.existsSync(path.join(PROJECT_ROOT, "memory")) &&
        fs.readdirSync(path.join(PROJECT_ROOT, "memory")).filter((f: string) => f.endsWith(".jsonl")).length > 0,
      identityModel: fs.existsSync(path.join(PROJECT_ROOT, "models", "identity", "config.json")),
      interactionMap: fs.existsSync(path.join(PROJECT_ROOT, "memory", "interaction_map_index.json")),
    },
    counts: {
      conversationFiles: fs.existsSync(path.join(PROJECT_ROOT, "conversations")) ? 
        fs.readdirSync(path.join(PROJECT_ROOT, "conversations")).filter((f: string) => f.endsWith(".jsonl")).length : 0,
      memoryFiles: fs.existsSync(path.join(PROJECT_ROOT, "memory")) ?
        fs.readdirSync(path.join(PROJECT_ROOT, "memory")).filter((f: string) => f.endsWith(".jsonl")).length : 0,
      files: fs.existsSync(path.join(PROJECT_ROOT, "files")) ?
        fs.readdirSync(path.join(PROJECT_ROOT, "files"), { recursive: true }).length : 0,
    }
  };
  
  return status;
}

export async function handleDataUploadConversations({ data }: { data: string | any }): Promise<{ success: boolean; message: string; path: string }> {
  const conversationsDir = path.join(PROJECT_ROOT, "conversations");
  fs.mkdirSync(conversationsDir, { recursive: true });
  
  const filePath = path.join(conversationsDir, "conversations.json");
  fs.writeFileSync(filePath, typeof data === "string" ? data : JSON.stringify(data, null, 2));
  
  return { success: true, message: "conversations.json uploaded successfully", path: filePath };
}

export async function handleDataUploadMemories({ data }: { data: string | any }): Promise<{ success: boolean; message: string; path: string }> {
  const memoryDir = path.join(PROJECT_ROOT, "memory");
  fs.mkdirSync(memoryDir, { recursive: true });
  
  const filePath = path.join(memoryDir, "memories.json");
  fs.writeFileSync(filePath, typeof data === "string" ? data : JSON.stringify(data, null, 2));
  
  return { success: true, message: "memories.json uploaded successfully", path: filePath };
}

export async function handleDataClean({ directory }: { directory: string }): Promise<{ success: boolean; message: string; deletedCount: number }> {
  const allowedDirs = ["conversations", "memory", "models", "training_data", "adapters"];
  
  if (!allowedDirs.includes(directory)) {
    throw new Error("Invalid directory");
  }
  
  const targetDir = path.join(PROJECT_ROOT, directory);
  
  if (!fs.existsSync(targetDir)) {
    return { success: true, message: `Directory ${directory} does not exist`, deletedCount: 0 };
  }
  
  const filesToKeep = directory === "conversations" ? ["conversations.json"] : 
                     directory === "memory" ? ["memories.json"] : [];
  
  const files = fs.readdirSync(targetDir);
  let deletedCount = 0;
  
  for (const file of files) {
    if (!filesToKeep.includes(file)) {
      const filePath = path.join(targetDir, file);
      const stat = fs.statSync(filePath);
      
      if (stat.isDirectory()) {
        fs.rmSync(filePath, { recursive: true, force: true });
        deletedCount++;
      } else {
        fs.unlinkSync(filePath);
        deletedCount++;
      }
    }
  }
  
  return { success: true, message: `Cleaned ${directory}: removed ${deletedCount} items`, deletedCount };
}

export async function handleDataDeleteSource({ type }: { type: string }): Promise<{ success: boolean; message: string }> {
  if (type !== "conversations" && type !== "memories") {
    return { success: false, message: "Invalid type. Must be 'conversations' or 'memories'" };
  }
  
  const filename = type === "conversations" ? "conversations.json" : "memories.json";
  const directory = type === "conversations" ? "conversations" : "memory";
  const filePath = path.join(PROJECT_ROOT, directory, filename);
  
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
      return { success: true, message: `${filename} deleted successfully` };
    } else {
      return { success: false, message: `${filename} not found` };
    }
  } catch (error: any) {
    return { success: false, message: error.message || `Failed to delete ${filename}` };
  }
}

// ============================================================================
// Conversations
// ============================================================================

export interface Conversation {
  id: string;
  filename: string;
  messageCount: number;
  firstDate: string | null;
  lastDate: string | null;
  title: string;
}

export async function handleDataConversationsList(): Promise<{ conversations: Conversation[] }> {
  const conversationsDir = path.join(PROJECT_ROOT, "conversations");
  
  if (!fs.existsSync(conversationsDir)) {
    return { conversations: [] };
  }
  
  const files = fs.readdirSync(conversationsDir)
    .filter((f: string) => f.endsWith(".jsonl") && f.startsWith("conversation_"))
    .map((f: string) => {
      const id = f.replace("conversation_", "").replace(".jsonl", "");
      const filePath = path.join(conversationsDir, f);
      const content = fs.readFileSync(filePath, "utf8");
      const lines = content.split("\n").filter((l: string) => l.trim());
      const messageCount = lines.length;
      
      let firstDate = null;
      let lastDate = null;
      let title = `Conversation ${id}`;
      
      if (lines.length > 0) {
        try {
          const first = JSON.parse(lines[0]);
          const last = JSON.parse(lines[lines.length - 1]);
          firstDate = first.timestamp || first.date;
          lastDate = last.timestamp || last.date;
          
          const firstUser = lines.find((l: string) => {
            const msg = JSON.parse(l);
            return msg.role === "user";
          });
          if (firstUser) {
            const msg = JSON.parse(firstUser);
            title = msg.content?.slice(0, 60) || title;
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
      
      return { id, filename: f, messageCount, firstDate, lastDate, title };
    });
  
  return { conversations: files };
}

export async function handleDataConversationGet({ id }: { id: string }): Promise<{ id: string; content: string; filePath: string }> {
  const filePath = path.join(PROJECT_ROOT, "conversations", `conversation_${id}.jsonl`);
  
  if (!fs.existsSync(filePath)) {
    throw new Error("Conversation not found");
  }
  
  const content = fs.readFileSync(filePath, "utf8");
  return { id, content, filePath };
}

export async function handleDataConversationUpdate({ id, content }: { id: string; content: string }): Promise<{ success: boolean; message: string; id: string }> {
  const filePath = path.join(PROJECT_ROOT, "conversations", `conversation_${id}.jsonl`);
  fs.writeFileSync(filePath, content, "utf8");
  
  return { success: true, message: "Conversation updated", id };
}

// ============================================================================
// Memories
// ============================================================================

export interface MemoryRecord {
  _file: string;
  _preview: string;
  [key: string]: any;
}

export async function handleDataMemoriesList(): Promise<{ memories: MemoryRecord[]; count: number }> {
  const memoryDir = path.join(PROJECT_ROOT, "memory");
  
  if (!fs.existsSync(memoryDir)) {
    return { memories: [], count: 0 };
  }
  
  const files = fs.readdirSync(memoryDir).filter((f: string) => f.endsWith(".jsonl"));
  const allRecords: MemoryRecord[] = [];
  
  for (const file of files) {
    const filePath = path.join(memoryDir, file);
    const content = fs.readFileSync(filePath, "utf8");
    const lines = content.split("\n").filter((l: string) => l.trim());
    
    for (const line of lines) {
      try {
        const record = JSON.parse(line);
        allRecords.push({
          ...record,
          _file: file,
          _preview: JSON.stringify(record).slice(0, 100),
        });
      } catch (e) {
        // Skip invalid lines
      }
    }
  }
  
  return { memories: allRecords, count: allRecords.length };
}

export async function handleDataMemoryFileGet({ filename }: { filename: string }): Promise<{ filename: string; content: string; filePath: string }> {
  if (!filename.endsWith(".jsonl") || filename.includes("..") || filename.includes("/")) {
    throw new Error("Invalid filename");
  }
  
  const filePath = path.join(PROJECT_ROOT, "memory", filename);
  
  if (!fs.existsSync(filePath)) {
    throw new Error("File not found");
  }
  
  const content = fs.readFileSync(filePath, "utf8");
  return { filename, content, filePath };
}

export async function handleDataMemoryFileUpdate({ filename, content }: { filename: string; content: string }): Promise<{ success: boolean; message: string; filename: string }> {
  if (!filename.endsWith(".jsonl") || filename.includes("..") || filename.includes("/")) {
    throw new Error("Invalid filename");
  }
  
  const filePath = path.join(PROJECT_ROOT, "memory", filename);
  fs.writeFileSync(filePath, content, "utf8");
  
  return { success: true, message: "Memory file updated", filename };
}

