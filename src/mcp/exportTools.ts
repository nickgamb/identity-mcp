/**
 * Export tools for backing up data
 */

import { listMemoryFiles, readAllRecords, MemoryFileName } from "../services/fileStore";
import { ConversationLoader } from "../services/conversationLoader";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";
import { logger } from "../utils/logger";

export interface ExportMemoriesRequest {
  files?: MemoryFileName[];
  outputPath?: string;
  format?: "jsonl" | "json";
}

export interface ExportMemoriesResponse {
  success: boolean;
  outputPath: string;
  recordCount: number;
  filesExported: number;
}

export interface ExportConversationsRequest {
  outputPath?: string;
  format?: "jsonl" | "json";
  limit?: number;
}

export interface ExportConversationsResponse {
  success: boolean;
  outputPath: string;
  conversationCount: number;
}

export async function handleExportMemories(
  req: ExportMemoriesRequest
): Promise<ExportMemoriesResponse> {
  try {
    const targetFiles: MemoryFileName[] = req.files && req.files.length > 0 
      ? req.files 
      : listMemoryFiles();
    const format = req.format || "jsonl";
    const outputPath = req.outputPath || join(process.cwd(), "exports", `memories-${Date.now()}.${format}`);
    
    // Ensure output directory exists
    const outputDir = outputPath.substring(0, outputPath.lastIndexOf("/") || outputPath.lastIndexOf("\\"));
    if (outputDir && !existsSync(outputDir)) {
      await mkdir(outputDir, { recursive: true });
    }
    
    let totalRecords = 0;
    const allRecords: any[] = [];
    
    for (const file of targetFiles) {
      try {
        const records = await readAllRecords(file);
        totalRecords += records.length;
        
        for (const record of records) {
          allRecords.push({
            ...record,
            _sourceFile: file,
          });
        }
      } catch (error) {
        logger.warn("Error exporting memory file", { file, error });
      }
    }
    
    // Write to file
    if (format === "jsonl") {
      const lines = allRecords.map(r => JSON.stringify(r)).join("\n");
      await writeFile(outputPath, lines, "utf8");
    } else {
      await writeFile(outputPath, JSON.stringify(allRecords, null, 2), "utf8");
    }
    
    logger.info("Exported memories", { outputPath, recordCount: totalRecords, filesExported: targetFiles.length });
    
    return {
      success: true,
      outputPath,
      recordCount: totalRecords,
      filesExported: targetFiles.length,
    };
  } catch (error) {
    logger.error("Error exporting memories", error);
    throw error;
  }
}

export async function handleExportConversations(
  req: ExportConversationsRequest
): Promise<ExportConversationsResponse> {
  try {
    const format = req.format || "jsonl";
    const limit = req.limit;
    const outputPath = req.outputPath || join(process.cwd(), "exports", `conversations-${Date.now()}.${format}`);
    
    // Ensure output directory exists
    const outputDir = outputPath.substring(0, outputPath.lastIndexOf("/") || outputPath.lastIndexOf("\\"));
    if (outputDir && !existsSync(outputDir)) {
      await mkdir(outputDir, { recursive: true });
    }
    
    const loader = new ConversationLoader();
    let conversations = await loader.loadAllConversations();
    
    if (limit && limit > 0) {
      conversations = conversations.slice(0, limit);
    }
    
    // Write to file
    if (format === "jsonl") {
      const lines = conversations.map(c => JSON.stringify(c)).join("\n");
      await writeFile(outputPath, lines, "utf8");
    } else {
      await writeFile(outputPath, JSON.stringify(conversations, null, 2), "utf8");
    }
    
    logger.info("Exported conversations", { outputPath, conversationCount: conversations.length });
    
    return {
      success: true,
      outputPath,
      conversationCount: conversations.length,
    };
  } catch (error) {
    logger.error("Error exporting conversations", error);
    throw error;
  }
}

