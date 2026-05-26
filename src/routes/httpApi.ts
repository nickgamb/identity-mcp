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
import { optionalAuth } from "../middleware/auth";
import { getUserContext } from "../utils/userContext";
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
  startPipelineRun,
  handlePipelineRunAll,
  handlePipelineStatus,
  handlePipelineListRunning,
  handlePipelineStream,
  handlePipelineOutput,
  handlePipelineStop,
} from "../mcp/pipelineTools";
import {
  handleIdentityAnalysisSummary,
  handleIdentityGetMomentum,
  handleIdentityGetNamingEvents,
  handleIdentityGetClusters,
  handleIdentityGetRelational,
} from "../mcp/identityAnalysisTools";
import {
  handleInteractionGetSummary,
  handleInteractionGetEvents,
  handleInteractionSearch,
  handleInteractionGetByTopic,
  handleInteractionGetTimeline,
} from "../mcp/interactionTools";
import {
  handleIdentityModelStatus,
  handleIdentityVerify,
  handleIdentityVerifyConversation,
  handleIdentityProfileSummary,
} from "../mcp/identityVerificationTools";
import {
  handleEegModelStatus,
  handleEegEnroll,
  handleEegAuthorize,
  handleEegProfileSummary,
} from "../mcp/eegIdentityTools";
import {
  handleSemanticSearch,
} from "../mcp/semanticSearchTools";
import {
  getLettaStatus,
  getLettaCoreMemory,
  updateLettaCoreMemory,
  getLettaArchival,
  getLettaMessages,
  updateLettaConfig,
  listOllamaModels,
  type ArchivalTypeFilter,
} from "../mcp/lettaProxy";
import {
  getReverieStatus,
  updateReverieConfig,
  getReveriePrompts,
  updateReveriePrompts,
} from "../services/reverieService";
import {
  getBackupStatus,
  updateBackupConfig,
  triggerBackupNow,
} from "../services/backupService";
import {
  handleDataStatus,
  handleDataUploadConversations,
  handleDataUploadMemories,
  handleDataUploadAnthropicConversations,
  handleDataUploadAnthropicMemories,
  handleDataClean,
  handleDataDeleteSource,
  handleDataConversationsList,
  handleDataConversationGet,
  handleDataConversationUpdate,
  handleDataMemoriesList,
  handleDataMemoryFileGet,
  handleDataMemoryFileUpdate,
} from "../mcp/dataManagementTools";

export const mcpRouter = Router();

// Apply optional authentication middleware to all routes
// This allows backward compatibility (anonymous access) while supporting OIDC when enabled
mcpRouter.use(optionalAuth);

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
    const userContext = getUserContext(req);
    const parsed = memoryListSchema.parse(req.body);
    const payload: MemoryListRequest = {
      files: parsed.files as MemoryFileName[] | undefined,
    };
    const result = await handleMemoryList(payload, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/memory.get", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const parsed = memoryGetSchema.parse(req.body);
    const payload: MemoryGetRequest = parsed as MemoryGetRequest;
    const result = await handleMemoryGet(payload, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/memory.append", async (req: Request, res: Response) => {
  try {
    const parsed = memoryAppendSchema.parse(req.body);
    const payload: MemoryAppendRequest = parsed as MemoryAppendRequest;
    const userContext = getUserContext(req);
    const result = await handleMemoryAppend(payload, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.get_core", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityGetCore(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.get_full", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityGetFull(userContext?.userId);
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
    const userContext = getUserContext(req);
    const parsed = memoryParseSchema.parse(req.body);
    const result = await handleMemoryParse(parsed, userContext?.userId);
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
    const userContext = getUserContext(req);
    const parsed = fileListSchema.parse(req.body);
    const payload: FileListRequest = parsed;
    const result = await handleFileList(payload, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.get", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const parsed = fileGetSchema.parse(req.body);
    const payload: FileGetRequest = parsed;
    const result = await handleFileGet(payload, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.search", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const parsed = fileSearchSchema.parse(req.body);
    const payload: FileSearchRequest = parsed;
    const result = await handleFileSearch(payload, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.numbered", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      folder: z.string().optional(),
      maxNumber: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileGetNumbered({ ...parsed, maxNumber: parsed.maxNumber ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.upload", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      filename: z.string(),
      content: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileUpload(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/file.delete", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      filepath: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFileDelete(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Memory Search
mcpRouter.post("/mcp/memory.search", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      query: z.string(),
      files: z.array(z.string()).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleMemorySearch({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
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
    const userContext = getUserContext(req);
    const parsed = schema.parse(req.body);
    const result = await handleConversationList({ ...parsed, limit: parsed.limit ?? undefined, offset: parsed.offset ?? undefined }, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.get", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      conversationId: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationGet(parsed, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.search", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      query: z.string(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationSearch({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/conversation.by_date_range", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      startDate: z.string().optional(),
      endDate: z.string().optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleConversationByDateRange({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId ?? null);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Statistics endpoints
mcpRouter.post("/mcp/memory.stats", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      files: z.array(z.string()).optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleMemoryStats(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/conversation.stats", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleConversationStats({}, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Unified Search
mcpRouter.post("/mcp/search.all", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      query: z.string(),
      sources: z.array(z.enum(["memories", "files", "conversations", "letta"])).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleUnifiedSearch({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Export endpoints
mcpRouter.post("/mcp/export.memories", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      files: z.array(z.string()).optional(),
      outputPath: z.string().optional(),
      format: z.enum(["jsonl", "json"]).optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleExportMemories(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/export.conversations", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      outputPath: z.string().optional(),
      format: z.enum(["jsonl", "json"]).optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleExportConversations({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Enhanced Fine-tuning endpoints
mcpRouter.post("/mcp/finetune.start", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      model_name: z.string().optional(),
      dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
      epochs: z.number().nullish(),
      learning_rate: z.number().nullish(),
      output_name: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneStart({ ...parsed, epochs: parsed.epochs ?? undefined, learning_rate: parsed.learning_rate ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.status", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      job_id: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneStatus(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/finetune.list", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleFinetuneList({}, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.cancel", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      job_id: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneCancel(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/finetune.export_dataset", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
      output_path: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleFinetuneExportDataset(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Identity Analysis endpoints
mcpRouter.get("/mcp/identity.analysis_summary", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityAnalysisSummary(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.momentum", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityGetMomentum(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.naming_events", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityGetNamingEvents({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.clusters", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      min_count: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityGetClusters({ ...parsed, min_count: parsed.min_count ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.relational", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityGetRelational(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Interaction Map endpoints (human communication patterns)
mcpRouter.get("/mcp/interaction.summary", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleInteractionGetSummary(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/interaction.events", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      event_type: z.string().optional(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleInteractionGetEvents({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/interaction.search", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      query: z.string(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleInteractionSearch({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/interaction.by_topic", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      topic: z.string(),
      limit: z.number().nullish(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleInteractionGetByTopic({ ...parsed, limit: parsed.limit ?? undefined }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/interaction.timeline", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      start_date: z.string().optional(),
      end_date: z.string().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleInteractionGetTimeline(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Identity Verification endpoints
mcpRouter.get("/mcp/identity.model_status", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityModelStatus(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.verify", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      message: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityVerify(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/identity.verify_conversation", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      messages: z.array(z.string()),
    });
    const parsed = schema.parse(req.body);
    const result = await handleIdentityVerifyConversation(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/identity.profile_summary", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleIdentityProfileSummary(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// EEG Identity Assurance endpoints
mcpRouter.get("/mcp/eeg.model_status", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleEegModelStatus(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/eeg.enroll", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      mode: z.enum(["auto", "synthetic", "hid"]).optional(),
      serial: z.string().optional(),
      task_duration: z.number().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEegEnroll(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/eeg.authorize", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      mode: z.enum(["auto", "synthetic", "hid"]).optional(),
      serial: z.string().optional(),
      window_seconds: z.number().optional(),
    });
    const parsed = schema.parse(req.body);
    const result = await handleEegAuthorize(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/eeg.profile_summary", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleEegProfileSummary(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Pipeline endpoints - for running processing scripts
mcpRouter.get("/mcp/pipeline.list", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handlePipelineList(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/pipeline.run", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const schema = z.object({
      script: z.string(),
      args: z.array(z.string()).optional(),
    });
    const parsed = schema.parse(req.body);
    // Return immediately; clients poll /mcp/pipeline.output/:id (long jobs can run for hours).
    const result = startPipelineRun(parsed, userContext?.userId);
    res.status(result.started ? 202 : 400).json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/pipeline.run_all", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handlePipelineRunAll(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// SSE stream for real-time pipeline output
mcpRouter.get("/mcp/pipeline.stream/:scriptId", (req: Request, res: Response) => {
  const { scriptId } = req.params;
  handlePipelineStream(scriptId, res);
});

// Poll pipeline output (JSON, proxy-friendly alternative to SSE)
mcpRouter.get("/mcp/pipeline.output/:scriptId", (req: Request, res: Response) => {
  const { scriptId } = req.params;
  const cursor = parseInt(req.query.cursor as string) || 0;
  res.json(handlePipelineOutput(scriptId, cursor));
});

// Stop a running pipeline script
mcpRouter.post("/mcp/pipeline.stop", async (req: Request, res: Response) => {
  try {
    const schema = z.object({ script: z.string() });
    const parsed = schema.parse(req.body);
    const result = await handlePipelineStop(parsed);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// ============================================================================
// Data Management Endpoints
// ============================================================================

// Check data status
mcpRouter.get("/mcp/data.status", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleDataStatus(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload conversations.json
mcpRouter.post("/mcp/data.upload_conversations", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }
    
    const result = await handleDataUploadConversations({ data }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload memories.json
mcpRouter.post("/mcp/data.upload_memories", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }
    
    const result = await handleDataUploadMemories({ data }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload Claude/Anthropic conversations.json
mcpRouter.post("/mcp/data.upload_anthropic_conversations", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }

    const result = await handleDataUploadAnthropicConversations({ data }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Upload Claude/Anthropic memories.json
mcpRouter.post("/mcp/data.upload_anthropic_memories", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { data } = req.body;
    if (!data) {
      res.status(400).json({ error: "No data provided" });
      return;
    }

    const result = await handleDataUploadAnthropicMemories({ data }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Semantic search via Letta archival memory
mcpRouter.post("/mcp/search.semantic", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { query, limit } = req.body;
    if (!query) {
      res.status(400).json({ error: "No query provided" });
      return;
    }
    const result = await handleSemanticSearch({ query, limit }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Clean directory
mcpRouter.post("/mcp/data.clean", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { directory } = req.body;
    const allowedDirs = ["conversations", "memory", "models_identity", "models_eeg", "training_data", "adapters"];
    
    if (!directory || !allowedDirs.includes(directory)) {
      res.status(400).json({ error: "Invalid directory" });
      return;
    }
    
    const result = await handleDataClean({ directory }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/data.delete_source", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { type } = req.body;
    const result = await handleDataDeleteSource({ type }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// List conversations
mcpRouter.get("/mcp/data.conversations", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleDataConversationsList(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Get specific conversation
mcpRouter.get("/mcp/data.conversation/:id", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { id } = req.params;
    const result = await handleDataConversationGet({ id }, userContext?.userId);
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
    const userContext = getUserContext(req);
    const { id } = req.params;
    const { content } = req.body;
    
    if (!content) {
      res.status(400).json({ error: "No content provided" });
      return;
    }
    
    const result = await handleDataConversationUpdate({ id, content }, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// List memory records
mcpRouter.get("/mcp/data.memories_list", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handleDataMemoriesList(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// Get memory file content
mcpRouter.get("/mcp/data.memory_file/:filename", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const { filename } = req.params;
    const result = await handleDataMemoryFileGet({ filename }, userContext?.userId);
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
    const userContext = getUserContext(req);
    const { filename } = req.params;
    const { content } = req.body;
    
    if (!content) {
      res.status(400).json({ error: "No content provided" });
      return;
    }
    
    const result = await handleDataMemoryFileUpdate({ filename, content }, userContext?.userId);
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
    const userContext = getUserContext(req);
    const schema = z.object({
      script: z.string(),
    });
    const parsed = schema.parse(req.body);
    const result = await handlePipelineStatus(parsed, userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/pipeline.running", async (req: Request, res: Response) => {
  try {
    const userContext = getUserContext(req);
    const result = await handlePipelineListRunning(userContext?.userId);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// ── Letta Memory System ─────────────────────────────────────────────────

mcpRouter.get("/mcp/ollama.models", async (_req: Request, res: Response) => {
  try {
    const result = await listOllamaModels();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/letta.status", async (_req: Request, res: Response) => {
  try {
    const result = await getLettaStatus();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/letta.memory", async (_req: Request, res: Response) => {
  try {
    const result = await getLettaCoreMemory();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/letta.memory.update", async (req: Request, res: Response) => {
  try {
    const { blockLabel, value } = req.body;
    if (!blockLabel || value === undefined) {
      res.status(400).json({ error: "blockLabel and value are required" });
      return;
    }
    const result = await updateLettaCoreMemory(blockLabel, value);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

const ARCHIVAL_TYPE_FILTERS = new Set([
  "conversation",
  "file",
  "memory",
  "analysis",
  "other",
]);

mcpRouter.get("/mcp/letta.archival", async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const cursor = req.query.cursor as string | undefined;
    const sort = req.query.sort === "newest" ? "newest" : "oldest";
    const typeRaw = req.query.type as string | undefined;
    const typeFilter: ArchivalTypeFilter | undefined =
      typeRaw && ARCHIVAL_TYPE_FILTERS.has(typeRaw)
        ? (typeRaw as ArchivalTypeFilter)
        : undefined;
    const dateFrom = (req.query.dateFrom as string) || undefined;
    const dateTo = (req.query.dateTo as string) || undefined;
    const result = await getLettaArchival(
      limit,
      cursor,
      sort,
      typeFilter,
      dateFrom,
      dateTo
    );
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/letta.messages", async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const cursor = req.query.cursor as string | undefined;
    const result = await getLettaMessages(limit, cursor);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.patch("/mcp/letta.config", async (req: Request, res: Response) => {
  // Model swap can unload + warm-load large GGUFs (many minutes on P40).
  req.setTimeout(20 * 60 * 1000);
  res.setTimeout(20 * 60 * 1000);
  try {
    const patch = req.body;
    if (!patch || Object.keys(patch).length === 0) {
      res.status(400).json({ error: "No config fields provided" });
      return;
    }
    const result = await updateLettaConfig(patch);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// ── Reverie (background self-reflection) ────────────────────────────────

mcpRouter.get("/mcp/reverie.status", async (_req: Request, res: Response) => {
  try {
    const result = getReverieStatus();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.patch("/mcp/reverie.config", async (req: Request, res: Response) => {
  try {
    const patch = req.body;
    if (!patch || Object.keys(patch).length === 0) {
      res.status(400).json({ error: "No config fields provided" });
      return;
    }
    const result = updateReverieConfig(patch);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.get("/mcp/reverie.prompts", async (_req: Request, res: Response) => {
  try {
    const prompts = getReveriePrompts();
    res.json({ prompts });
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.put("/mcp/reverie.prompts", async (req: Request, res: Response) => {
  try {
    const { prompts } = req.body;
    if (!prompts) {
      res.status(400).json({ error: "No prompts provided" });
      return;
    }
    const result = updateReveriePrompts(prompts);
    if (!result.success) {
      res.status(400).json(result);
      return;
    }
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

// ── Backup ─────────────────────────────────────────────────────────────

mcpRouter.get("/mcp/backup.status", async (_req: Request, res: Response) => {
  try {
    const result = getBackupStatus();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.patch("/mcp/backup.config", async (req: Request, res: Response) => {
  try {
    const patch = req.body;
    if (!patch || Object.keys(patch).length === 0) {
      res.status(400).json({ error: "No config fields provided" });
      return;
    }
    const result = updateBackupConfig(patch);
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});

mcpRouter.post("/mcp/backup.trigger", async (req: Request, res: Response) => {
  // Large Letta DB dumps can take many minutes.
  req.setTimeout(20 * 60 * 1000);
  res.setTimeout(20 * 60 * 1000);
  try {
    const result = triggerBackupNow();
    res.json(result);
  } catch (err) {
    handleError(res, err);
  }
});


