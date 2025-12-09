/**
 * HTTP API Routes - Direct REST-style HTTP endpoints
 * 
 * This file provides direct HTTP API endpoints at /mcp/* for:
 * - Direct HTTP calls (curl, Postman, scripts)
 * - Simple JSON request/response format
 * - Non-MCP protocol clients
 * 
 * For MCP protocol clients (like LibreChat), see mcpProtocol.ts
 */

import { Router, Request, Response } from "express";
import { z } from "zod";
import { logger } from "../utils/logger";
import {
  handleMemoryAppend,
  handleMemoryGet,
  handleMemoryList,
} from "../mcp/memoryTools";
import {
  handleIdentityGetCore,
  handleIdentityGetFull,
} from "../mcp/identityTools";
import {
  handleMemoryParse,
} from "../mcp/memoryParserTools";
import {
  handleFileList,
  handleFileGet,
  handleFileSearch,
  handleFileGetNumbered,
  handleFileUpload,
  handleFileDelete,
  FileListRequest,
  FileGetRequest,
  FileSearchRequest,
  FileUploadRequest,
  FileDeleteRequest,
} from "../mcp/fileTools";
import {
  MemoryAppendRequest,
  MemoryGetRequest,
  MemoryListRequest,
  MemoryFileName,
} from "../mcp/types";
import {
  handleConversationList,
  handleConversationGet,
  handleConversationSearch,
  handleConversationByDateRange,
} from "../mcp/conversationTools";
import {
  handleMemorySearch,
} from "../mcp/memorySearchTools";
import {
  handleMemoryStats,
  handleConversationStats,
} from "../mcp/statisticsTools";
import {
  handleUnifiedSearch,
} from "../mcp/unifiedSearchTools";
import {
  handleExportMemories,
  handleExportConversations,
} from "../mcp/exportTools";
import {
  handleFinetuneStart,
  handleFinetuneStatus,
  handleFinetuneList,
  handleFinetuneCancel,
  handleFinetuneExportDataset,
} from "../mcp/finetuneTools";
import {
  handlePipelineList,
  handlePipelineRun,
  handlePipelineRunAll,
  handlePipelineStatus,
  handlePipelineListRunning,
} from "../mcp/pipelineTools";
import {
  handleIdentityAnalysisSummary,
  handleIdentityGetMomentum,
  handleIdentityGetNamingEvents,
  handleIdentityGetClusters,
  handleIdentityGetRelational,
} from "../mcp/identityAnalysisTools";
import {
  handleEmergenceGetSummary,
  handleEmergenceGetEvents,
  handleEmergenceSearch,
  handleEmergenceGetSymbolicConversations,
  handleEmergenceGetTimeline,
} from "../mcp/emergenceTools";
import {
  handleIdentityModelStatus,
  handleIdentityVerify,
  handleIdentityVerifyConversation,
  handleIdentityProfileSummary,
} from "../mcp/identityVerificationTools";
import {
  handleDataStatus,
  handleDataUploadConversations,
  handleDataUploadMemories,
  handleDataClean,
  handleDataConversationsList,
  handleDataConversationGet,
  handleDataConversationUpdate,
  handleDataMemoriesList,
  handleDataMemoryFileGet,
  handleDataMemoryFileUpdate,
} from "../mcp/dataManagementTools";

export const mcpRouter = Router();

const memoryListSchema = z.object({
  files: z.array(z.string()).optional(),
});

const memoryGetSchema = z.object({
  file: z.string(),
  filters: z
    .object({
      type: z.string().optional(),
      tags: z.array(z.string()).optional(),
    })
    .optional(),
  limit: z.number().int().positive().nullish(),
});

const memoryAppendSchema = z.object({
  file: z.string(),
  record: z.record(z.any()),
});

function handleError(res: Response, error: unknown) {
  logger.error("MCP route error", error);
  res.status(400).json({
    error: "bad_request",
    message: error instanceof Error ? error.message : String(error),
  });
}

mcpRouter.post("/mcp/memory.list", async (req: Request, res: Response) => {
  try {
    const parsed = memoryListSchema.parse(req.body);
    const payload: MemoryListRequest = {
      files: parsed.files as MemoryFileName[] | undefined,
    };
    const result = await handleMemoryList(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/memory.get", async (req: Request, res: Response) => {
  try {
    const parsed = memoryGetSchema.parse(req.body);
    const payload: MemoryGetRequest = parsed as MemoryGetRequest;
    const result = await handleMemoryGet(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/memory.append", async (req: Request, res: Response) => {
  try {
    const parsed = memoryAppendSchema.parse(req.body);
    const payload: MemoryAppendRequest = parsed as MemoryAppendRequest;
    const result = await handleMemoryAppend(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.get_core", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityGetCore();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.get_full", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityGetFull();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});


// Memory parser endpoints
const memoryParseSchema = z.object({
  force: z.boolean().optional(),
});

mcpRouter.post("/mcp/memory.parse", async (req: Request, res: Response) => {
  try {
    const parsed = memoryParseSchema.parse(req.body);
    const result = await handleMemoryParse(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// File/RAG endpoints - for accessing raw files that memories point to
const fileListSchema = z.object({
  folder: z.string().optional(), // Filter to specific folder
  category: z.string().optional(),
});

const fileGetSchema = z.object({
  filepath: z.string(), // Relative path from files directory
});

const fileSearchSchema = z.object({
  query: z.string(),
  folder: z.string().optional(),
});

mcpRouter.post("/mcp/file.list", async (req: Request, res: Response) => {
  try {
    const parsed = fileListSchema.parse(req.body);
    const payload: FileListRequest = parsed;
    const result = await handleFileList(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.get", async (req: Request, res: Response) => {
  try {
    const parsed = fileGetSchema.parse(req.body);
    const payload: FileGetRequest = parsed;
    const result = await handleFileGet(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.search", async (req: Request, res: Response) => {
  try {
    const parsed = fileSearchSchema.parse(req.body);
    const payload: FileSearchRequest = parsed;
    const result = await handleFileSearch(payload);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.numbered", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      folder: z.string().optional(),
      maxNumber: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileGetNumbered(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.upload", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      filename: z.string(),
      content: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileUpload(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.delete", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      filepath: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileDelete(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Memory Search
mcpRouter.post("/mcp/memory.search", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      query: z.string(),
      files: z.array(z.string()).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleMemorySearch(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Conversation endpoints
mcpRouter.post("/mcp/conversation.list", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      limit: z.number().nullish(),
      offset: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationList(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.get", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      conversationId: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationGet(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.search", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      query: z.string(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationSearch(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.by_date_range", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      startDate: z.string().optional(),
      endDate: z.string().optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationByDateRange(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Statistics endpoints
mcpRouter.post("/mcp/memory.stats", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      files: z.array(z.string()).optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleMemoryStats(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/conversation.stats", async (_req: Request, res: Response) => {
  try {
    const result = await handleConversationStats({});
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Unified Search
mcpRouter.post("/mcp/search.all", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      query: z.string(),
      sources: z.array(z.enum(["memories", "files", "conversations"])).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleUnifiedSearch(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Export endpoints
mcpRouter.post("/mcp/export.memories", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      files: z.array(z.string()).optional(),
      outputPath: z.string().optional(),
      format: z.enum(["jsonl", "json"]).optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleExportMemories(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/export.conversations", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      outputPath: z.string().optional(),
      format: z.enum(["jsonl", "json"]).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleExportConversations(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Enhanced Fine-tuning endpoints
mcpRouter.post("/mcp/finetune.start", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      model_name: z.string().optional(),
      dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
      epochs: z.number().nullish(),
      learning_rate: z.number().nullish(),
      output_name: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneStart(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.status", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      job_id: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneStatus(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/finetune.list", async (_req: Request, res: Response) => {
  try {
    const result = await handleFinetuneList({});
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.cancel", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      job_id: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneCancel(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.export_dataset", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
      output_path: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneExportDataset(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Identity Analysis endpoints
mcpRouter.get("/mcp/identity.analysis_summary", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityAnalysisSummary();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.momentum", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityGetMomentum();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.naming_events", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityGetNamingEvents(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.clusters", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      min_count: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityGetClusters(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.relational", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityGetRelational();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Emergence endpoints
mcpRouter.get("/mcp/emergence.summary", async (_req: Request, res: Response) => {
  try {
    const result = await handleEmergenceGetSummary();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/emergence.events", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      event_type: z.string().optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEmergenceGetEvents(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/emergence.search", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      query: z.string(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEmergenceSearch(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/emergence.symbolic_conversations", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEmergenceGetSymbolicConversations(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/emergence.timeline", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      start_date: z.string().optional(),
      end_date: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEmergenceGetTimeline(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Identity Verification endpoints
mcpRouter.get("/mcp/identity.model_status", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityModelStatus();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.verify", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      message: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityVerify(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.verify_conversation", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      messages: z.array(z.string()),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityVerifyConversation(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.profile_summary", async (_req: Request, res: Response) => {
  try {
    const result = await handleIdentityProfileSummary();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Pipeline endpoints - for running processing scripts
mcpRouter.get("/mcp/pipeline.list", async (_req: Request, res: Response) => {
  try {
    const result = await handlePipelineList();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/pipeline.run", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      script: z.string(),
      args: z.array(z.string()).optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handlePipelineRun(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/pipeline.run_all", async (_req: Request, res: Response) => {
  try {
    const result = await handlePipelineRunAll();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// ============================================================================
// Data Management Endpoints
// ============================================================================

// Check data status
mcpRouter.get("/mcp/data.status", async (_req: Request, res: Response) => {
  try {
    const result = await handleDataStatus();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload conversations.json
mcpRouter.post("/mcp/data.upload_conversations", async (req: Request, res: Response) => {
  try {
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }
    
    const result = await handleDataUploadConversations({ data });
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload memories.json
mcpRouter.post("/mcp/data.upload_memories", async (req: Request, res: Response) => {
  try {
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }
    
    const result = await handleDataUploadMemories({ data });
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Clean directory
mcpRouter.post("/mcp/data.clean", async (req: Request, res: Response) => {
  try {
    const { directory } = req.body;
    const allowedDirs = ["conversations", "memory", "models", "training_data", "adapters"];
    
    if (!directory || !allowedDirs.includes(directory)) {
      res.status(400).json({ error: "Invalid directory" });
      return;
    }
    
    const result = await handleDataClean({ directory });
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// List conversations
mcpRouter.get("/mcp/data.conversations", async (_req: Request, res: Response) => {
  try {
    const result = await handleDataConversationsList();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Get specific conversation
mcpRouter.get("/mcp/data.conversation/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const result = await handleDataConversationGet({ id });
    res.json(result);
  } catch (err) {
    if (err instanceof Error && err.message === "Conversation not found") {
      res.status(404).json({ error: err.message });
    } else {
      handleError(res, err);
    }
  }
});

// Update conversation
mcpRouter.post("/mcp/data.conversation/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const { content } = req.body;
    
    if (!content) {
      res.status(400).json({ error: "No content provided" });
      return;
    }
    
    const result = await handleDataConversationUpdate({ id, content });
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// List memory records
mcpRouter.get("/mcp/data.memories_list", async (_req: Request, res: Response) => {
  try {
    const result = await handleDataMemoriesList();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Get memory file content
mcpRouter.get("/mcp/data.memory_file/:filename", async (req: Request, res: Response) => {
  try {
    const { filename } = req.params;
    const result = await handleDataMemoryFileGet({ filename });
    res.json(result);
  } catch (err) {
    if (err instanceof Error && (err.message === "Invalid filename" || err.message === "File not found")) {
      res.status(err.message === "Invalid filename" ? 400 : 404).json({ error: err.message });
    } else {
      handleError(res, err);
    }
  }
});

// Update memory file
mcpRouter.post("/mcp/data.memory_file/:filename", async (req: Request, res: Response) => {
  try {
    const { filename } = req.params;
    const { content } = req.body;
    
    if (!content) {
      res.status(400).json({ error: "No content provided" });
      return;
    }
    
    const result = await handleDataMemoryFileUpdate({ filename, content });
    res.json(result);
  } catch (err) {
    if (err instanceof Error && err.message === "Invalid filename") {
      res.status(400).json({ error: err.message });
    } else {
      handleError(res, err);
    }
  }
});

mcpRouter.post("/mcp/pipeline.status", async (req: Request, res: Response) => {
  try {
    const schema = z.object({
      script: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handlePipelineStatus(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/pipeline.running", async (_req: Request, res: Response) => {
  try {
    const result = await handlePipelineListRunning();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});


