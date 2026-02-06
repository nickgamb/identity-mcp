/**
 * Streamable HTTP MCP endpoint for Identity MCP.
 * Uses the official MCP SDK so LibreChat can stream tools over SSE.
 */

import { Router, Request, Response } from "express";
import { randomUUID } from "crypto";
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { logger } from "../utils/logger";
import { optionalAuth } from "../middleware/auth";
import { getUserContext, getRequiredUserId } from "../utils/userContext";
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
} from "../mcp/fileTools";
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
  handlePipelineList,
  handlePipelineRun,
  handlePipelineRunAll,
  handlePipelineStatus,
  handlePipelineListRunning,
} from "../mcp/pipelineTools";
import {
  handleEegModelStatus,
  handleEegEnroll,
  handleEegAuthorize,
  handleEegProfileSummary,
} from "../mcp/eegIdentityTools";
import {
  handleDataStatus,
  handleDataUploadConversations,
  handleDataUploadMemories,
  handleDataClean,
  handleDataDeleteSource,
  handleDataConversationsList,
  handleDataConversationGet,
  handleDataConversationUpdate,
  handleDataMemoriesList,
  handleDataMemoryFileGet,
  handleDataMemoryFileUpdate,
} from "../mcp/dataManagementTools";

export const mcpProtocolRouter = Router();

// Apply optional authentication middleware
mcpProtocolRouter.use(optionalAuth);

// Map of sessionId -> transport, following the official MCP SDK example.
// Each session gets its own transport and connected server instance.
const transports: Record<string, StreamableHTTPServerTransport> = {};

// Map of sessionId -> userId for multi-user support
const sessionUsers: Record<string, string | null> = {};

const toContent = (payload: unknown): { content: Array<{ type: "text"; text: string }> } => ({
  content: [
    {
      type: "text" as const,
      text: typeof payload === "string" ? payload : JSON.stringify(payload, null, 2),
    },
  ],
});


function registerTools(server: McpServer, getUserId: () => string | null) {
  server.registerTool(
    "memory_list",
    {
      title: "List Memory Files",
      description: "List memory files and record counts.",
      inputSchema: z.object({
        files: z.array(z.string()).optional(),
      }),
    },
    async ({ files }) => toContent(await handleMemoryList({ files }, getUserId())),
  );

  server.registerTool(
    "memory_get",
    {
      title: "Get Memory",
      description: "Retrieve records from a memory file with optional filters (type, tags, date range).",
      inputSchema: z.object({
        file: z.string(),
        filters: z
          .object({
            type: z.string().optional(),
            tags: z.array(z.string()).optional(),
            startDate: z.string().optional(),
            endDate: z.string().optional(),
          })
          .optional(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleMemoryGet(payload, getUserId())),
  );

  server.registerTool(
    "memory_search",
    {
      title: "Search Memories",
      description: "Full-text search across all memory files. Searches in record content, not just metadata.",
      inputSchema: z.object({
        query: z.string(),
        files: z.array(z.string()).optional(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleMemorySearch(payload, getUserId())),
  );

  server.registerTool(
    "memory_append",
    {
      title: "Append Memory",
      description: "Append a record to memory.",
      inputSchema: z.object({
        file: z.string(),
        record: z
          .object({
            id: z.string().optional(),
            type: z.string(),
          })
          .catchall(z.any()),
      }),
    },
    async ({ file, record }) => toContent(await handleMemoryAppend({ file, record }, getUserId())),
  );

  server.registerTool(
    "memory_parse",
    {
      title: "Reparse Memories",
      description: "Rebuild user.context.jsonl from memories.json.",
      inputSchema: z.object({
        force: z.boolean().optional(),
      }),
    },
    async (payload) => toContent(await handleMemoryParse(payload, getUserId())),
  );

  server.registerTool(
    "identity_get_core",
    {
      title: "Core Identity",
      description: "Retrieve breath, vows, and prime directives.",
      inputSchema: z.object({}).optional(),
    },
    async () => toContent(await handleIdentityGetCore(getUserId())),
  );

  server.registerTool(
    "identity_get_full",
    {
      title: "Full Identity Bundle",
      description: "Retrieve the complete Cathedral identity bundle.",
      inputSchema: z.object({}).optional(),
    },
    async () => toContent(await handleIdentityGetFull(getUserId())),
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // Identity Analysis Tools (from analyze_identity.py output)
  // ─────────────────────────────────────────────────────────────────────────────

  server.registerTool(
    "identity_analysis_summary",
    {
      title: "Identity Analysis Summary",
      description: "Get overview of identity pattern analysis including relational and self-referential patterns.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleIdentityAnalysisSummary(getUserId())),
  );

  server.registerTool(
    "identity_get_momentum",
    {
      title: "Get Pattern Momentum",
      description: "Get patterns that are rising or falling over time, showing identity evolution.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleIdentityGetMomentum(getUserId())),
  );

  server.registerTool(
    "identity_get_naming_events",
    {
      title: "Get Naming Events",
      description: "Get moments where names/identities were established in conversations.",
      inputSchema: z.object({
        limit: z.number().nullish().describe("Maximum events to return"),
      }),
    },
    async ({ limit }) => toContent(await handleIdentityGetNamingEvents({ limit }, getUserId())),
  );

  server.registerTool(
    "identity_get_clusters",
    {
      title: "Get Co-occurrence Clusters",
      description: "Get concepts that frequently appear together, revealing associative patterns.",
      inputSchema: z.object({
        min_count: z.number().nullish().describe("Minimum co-occurrence count"), 
      }),
    },
    async ({ min_count }) => toContent(await handleIdentityGetClusters({ min_count }, getUserId())),
  );

  server.registerTool(
    "identity_get_relational",
    {
      title: "Get Relational Patterns",
      description: "Get we/I ratios and role language patterns showing relationship dynamics.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleIdentityGetRelational(getUserId())),
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // Interaction Map Tools (from build_interaction_map.py output)
  // Focus: Human communication patterns and identity fingerprinting
  // ─────────────────────────────────────────────────────────────────────────────

  server.registerTool(
    "interaction_summary",
    {
      title: "Interaction Summary",
      description: "Get summary of interaction data including event counts, topic/tone distribution, and human message stats.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleInteractionGetSummary(getUserId())),
  );

  server.registerTool(
    "interaction_get_events",
    {
      title: "Get Key Communication Events",
      description: "Get key human communication events (problem-solving, tempo changes, topic transitions, tone shifts).",
      inputSchema: z.object({
        event_type: z.string().optional().describe("Filter by event type (PROBLEM_SOLVING, TEMPO_CHANGE, TOPIC_TRANSITION, TONE_SHIFT)"),
        limit: z.number().nullish().describe("Maximum events to return"),
      }),
    },
    async ({ event_type, limit }) => toContent(await handleInteractionGetEvents({ event_type, limit }, getUserId())),
  );

  server.registerTool(
    "interaction_search",
    {
      title: "Search Interaction Index",
      description: "Search conversations by topic, tone, or keyword in message content.",
      inputSchema: z.object({
        query: z.string().describe("Search query (topic, tone, or keyword)"),
        limit: z.number().nullish().describe("Maximum results to return"),
      }),
    },
    async ({ query, limit }) => toContent(await handleInteractionSearch({ query, limit }, getUserId())),
  );

  server.registerTool(
    "interaction_get_by_topic",
    {
      title: "Get Conversations by Topic",
      description: "Get conversations filtered by specific topic tag.",
      inputSchema: z.object({
        topic: z.string().describe("Topic tag (e.g., technical, emotional, creative, philosophical)"),
        limit: z.number().nullish().describe("Maximum conversations to return"),
      }),
    },
    async ({ topic, limit }) => toContent(await handleInteractionGetByTopic({ topic, limit }, getUserId())),
  );

  server.registerTool(
    "interaction_timeline",
    {
      title: "Get Event Timeline",
      description: "Get timeline of key human communication events, optionally filtered by date range.",
      inputSchema: z.object({
        start_date: z.string().optional().describe("Start date (YYYY-MM-DD)"),
        end_date: z.string().optional().describe("End date (YYYY-MM-DD)"),
      }),
    },
    async ({ start_date, end_date }) => toContent(await handleInteractionGetTimeline({ start_date, end_date }, getUserId())),
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // Identity Verification Tools (behavioral biometric verification)
  // ─────────────────────────────────────────────────────────────────────────────

  server.registerTool(
    "identity_model_status",
    {
      title: "Identity Model Status",
      description: "Check if the identity verification model is trained and available.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleIdentityModelStatus(getUserId())),
  );

  server.registerTool(
    "identity_verify",
    {
      title: "Verify Identity",
      description: "Verify if a message matches the trained identity profile. Returns confidence score.",
      inputSchema: z.object({
        message: z.string().describe("The message to verify against the identity profile"),
      }),
    },
    async ({ message }) => toContent(await handleIdentityVerify({ message }, getUserId())),
  );

  server.registerTool(
    "identity_verify_conversation",
    {
      title: "Verify Conversation",
      description: "Verify multiple messages against the identity profile. Returns overall confidence.",
      inputSchema: z.object({
        messages: z.array(z.string()).describe("Array of messages to verify"),
      }),
    },
    async ({ messages }) => toContent(await handleIdentityVerifyConversation({ messages }, getUserId())),
  );

  server.registerTool(
    "identity_profile_summary",
    {
      title: "Identity Profile Summary",
      description: "Get summary of the trained identity profile (stylistic features, vocabulary).",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleIdentityProfileSummary(getUserId())),
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // EEG Identity Assurance Tools (brainwave-based identity)
  // ─────────────────────────────────────────────────────────────────────────────

  server.registerTool(
    "eeg_model_status",
    {
      title: "EEG Identity Model Status",
      description: "Check if the EEG brainwave identity model is enrolled and available.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleEegModelStatus(getUserId())),
  );

  server.registerTool(
    "eeg_enroll",
    {
      title: "Enroll Brainwaves",
      description: "Run EEG brainwave enrollment to create a reference identity model. Guides through neurofeedback tasks while capturing EEG.",
      inputSchema: z.object({
        mode: z.enum(["synthetic", "hid"]).optional().describe("Connection mode: 'hid' for hardware, 'synthetic' for testing"),
        serial: z.string().optional().describe("EMOTIV device serial number (required for HID mode)"),
        task_duration: z.number().nullish().describe("Duration per task in seconds (default: 20)"),
      }),
    },
    async ({ mode, serial, task_duration }) => toContent(await handleEegEnroll({ mode, serial, task_duration: task_duration ?? undefined }, getUserId())),
  );

  server.registerTool(
    "eeg_authorize",
    {
      title: "Authorize Brainwaves",
      description: "Read live EEG and compare against enrolled brainwave model. Returns an assurance signal (0.0-1.0).",
      inputSchema: z.object({
        mode: z.enum(["synthetic", "hid"]).optional().describe("Connection mode: 'hid' for hardware, 'synthetic' for testing"),
        serial: z.string().optional().describe("EMOTIV device serial number (required for HID mode)"),
        window_seconds: z.number().nullish().describe("EEG capture window in seconds (default: 10)"),
      }),
    },
    async ({ mode, serial, window_seconds }) => toContent(await handleEegAuthorize({ mode, serial, window_seconds: window_seconds ?? undefined }, getUserId())),
  );

  server.registerTool(
    "eeg_profile_summary",
    {
      title: "EEG Profile Summary",
      description: "Get summary of the trained EEG identity profile including spectral data, enrollment tasks, and statistics.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleEegProfileSummary(getUserId())),
  );

  // ─────────────────────────────────────────────────────────────────────────────
  // Pipeline Tools - Run processing scripts
  // ─────────────────────────────────────────────────────────────────────────────

  server.registerTool(
    "pipeline_list",
    {
      title: "List Pipeline Scripts",
      description: "List available processing scripts that can be run.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handlePipelineList(getUserId())),
  );

  server.registerTool(
    "pipeline_run",
    {
      title: "Run Pipeline Script",
      description: "Run a specific processing script. Use pipeline_list to see available scripts.",
      inputSchema: z.object({
        script: z.string().describe("Script ID (e.g., 'parse_conversations', 'analyze_patterns')"),
        args: z.array(z.string()).optional().describe("Optional command-line arguments"),
      }),
    },
    async ({ script, args }) => toContent(await handlePipelineRun({ script, args }, getUserId())),
  );

  server.registerTool(
    "pipeline_run_all",
    {
      title: "Run Full Pipeline",
      description: "Run all processing scripts in order (parse → analyze → build). Stops on first failure.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handlePipelineRunAll(getUserId())),
  );

  server.registerTool(
    "pipeline_status",
    {
      title: "Check Pipeline Script Status",
      description: "Check if a specific script is currently running and get its status.",
      inputSchema: z.object({
        script: z.string().describe("Script ID to check status for"),
      }),
    },
    async ({ script }) => toContent(await handlePipelineStatus({ script }, getUserId())),
  );

  server.registerTool(
    "pipeline_list_running",
    {
      title: "List Running Scripts",
      description: "List all currently running pipeline scripts.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handlePipelineListRunning(getUserId())),
  );

  server.registerTool(
    "file_list",
    {
      title: "List Files",
      description: "List files from RAG folders.",
      inputSchema: z.object({
        folder: z.string().optional(),
        category: z.string().optional(),
      }),
    },
    async ({ folder, category }) => toContent(await handleFileList({ folder, category }, getUserId())),
  );

  server.registerTool(
    "file_get",
    {
      title: "Get File",
      description: "Retrieve a specific file by path.",
      inputSchema: z.object({
        filepath: z.string(),
      }),
    },
    async ({ filepath }) => toContent(await handleFileGet({ filepath }, getUserId())),
  );

  server.registerTool(
    "file_search",
    {
      title: "Search Files",
      description: "Full text search across all files in RAG storage.",
      inputSchema: z.object({
        query: z.string(),
        folder: z.string().optional(),
      }),
    },
    async ({ query, folder }) => toContent(await handleFileSearch({ query, folder }, getUserId())),
  );

  server.registerTool(
    "file_get_numbered",
    {
      title: "Get Numbered Files",
      description: "Get files numbered 001-N from files directory. Useful for core/foundation documents.",
      inputSchema: z.object({
        folder: z.string().optional(),
        maxNumber: z.number().nullish(),
      }),
    },
    async ({ folder, maxNumber }) => toContent(await handleFileGetNumbered({ folder, maxNumber }, getUserId())),
  );

  // Fine-tuning Tools
  server.registerTool(
    "finetune_start",
    {
      title: "Start LoRA Fine-Tuning",
      description: "Start a LoRA fine-tuning job using conversations, files, and memory. Trains the model on your corpus.\n\nParameters:\n- model_name (optional): Model to fine-tune (default: 'gpt-oss:20b')\n- dataset_source (optional): 'conversations', 'memories', 'files', or 'all' (default: 'all')\n- epochs (optional): Training epochs (default: 3)\n- learning_rate (optional): Learning rate (default: 2e-5)\n- output_name (optional): Name for the fine-tuned adapter\n\nThis is a long-running process (several hours). Use finetune_status to check progress.",
      inputSchema: z.object({
        model_name: z.string().optional(),
        dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
        epochs: z.number().nullish(),
        learning_rate: z.number().nullish(),
        output_name: z.string().optional(),
      }),
    },
    async (payload) => {
      return toContent(await handleFinetuneStart(payload, getUserId()));
    },
  );

  server.registerTool(
    "finetune_status",
    {
      title: "Check Fine-Tuning Status",
      description: "Check the status of a fine-tuning job.\n\nRequired: {\"job_id\": \"finetune-1234567890\"}\n\nReturns status: pending, running, completed, or failed. Also shows progress percentage and any messages.",
      inputSchema: z.object({
        job_id: z.string(),
      }),
    },
    async (payload) => {
      return toContent(await handleFinetuneStatus(payload, getUserId()));
    },
  );

  server.registerTool(
    "finetune_list",
    {
      title: "List Fine-Tuning Jobs",
      description: "List all fine-tuning jobs (active and completed).",
      inputSchema: z.object({}).optional(),
    },
    async () => toContent(await handleFinetuneList({}, getUserId())),
  );

  server.registerTool(
    "finetune_cancel",
    {
      title: "Cancel Fine-Tuning Job",
      description: "Cancel a running fine-tuning job.",
      inputSchema: z.object({
        job_id: z.string(),
      }),
    },
    async (payload) => toContent(await handleFinetuneCancel(payload, getUserId())),
  );

  server.registerTool(
    "finetune_export_dataset",
    {
      title: "Export Training Dataset",
      description: "Export training dataset without starting training. Useful for inspection or external training.",
      inputSchema: z.object({
        dataset_source: z.enum(["conversations", "memories", "files", "all"]).optional(),
        output_path: z.string().optional(),
      }),
    },
    async (payload) => toContent(await handleFinetuneExportDataset(payload, getUserId())),
  );

  // Conversation Tools
  server.registerTool(
    "conversation_list",
    {
      title: "List Conversations",
      description: "List all conversations with metadata (message count, date range).",
      inputSchema: z.object({
        limit: z.number().nullish(),
        offset: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleConversationList(payload, getUserId())),
  );

  server.registerTool(
    "conversation_get",
    {
      title: "Get Conversation",
      description: "Get a specific conversation by ID with all messages.",
      inputSchema: z.object({
        conversationId: z.string(),
      }),
    },
    async (payload) => toContent(await handleConversationGet(payload, getUserId())),
  );

  server.registerTool(
    "conversation_search",
    {
      title: "Search Conversations",
      description: "Search conversations by content. Returns conversations containing the query text.",
      inputSchema: z.object({
        query: z.string(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleConversationSearch(payload, getUserId())),
  );

  server.registerTool(
    "conversation_by_date_range",
    {
      title: "Get Conversations by Date Range",
      description: "Get conversations within a date range. Useful for temporal analysis.",
      inputSchema: z.object({
        startDate: z.string().optional(),
        endDate: z.string().optional(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleConversationByDateRange(payload, getUserId())),
  );

  // Statistics Tools
  server.registerTool(
    "memory_stats",
    {
      title: "Memory Statistics",
      description: "Get statistics about memory files (counts, types, tags, date ranges).",
      inputSchema: z.object({
        files: z.array(z.string()).optional(),
      }),
    },
    async (payload) => toContent(await handleMemoryStats(payload, getUserId())),
  );

  server.registerTool(
    "conversation_stats",
    {
      title: "Conversation Statistics",
      description: "Get statistics about conversations (total count, messages, date ranges, by year).",
      inputSchema: z.object({}).optional(),
    },
    async () => toContent(await handleConversationStats({}, getUserId())),
  );

  // Unified Search
  server.registerTool(
    "search_all",
    {
      title: "Unified Search",
      description: "Search across memories, files, and conversations simultaneously. Returns results from all sources.",
      inputSchema: z.object({
        query: z.string(),
        sources: z.array(z.enum(["memories", "files", "conversations"])).optional(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleUnifiedSearch(payload, getUserId())),
  );

  // Export Tools
  server.registerTool(
    "export_memories",
    {
      title: "Export Memories",
      description: "Export memory files to JSONL or JSON format for backup or analysis.",
      inputSchema: z.object({
        files: z.array(z.string()).optional(),
        outputPath: z.string().optional(),
        format: z.enum(["jsonl", "json"]).optional(),
      }),
    },
    async (payload) => toContent(await handleExportMemories(payload, getUserId())),
  );

  server.registerTool(
    "export_conversations",
    {
      title: "Export Conversations",
      description: "Export conversations to JSONL or JSON format for backup or analysis.",
      inputSchema: z.object({
        outputPath: z.string().optional(),
        format: z.enum(["jsonl", "json"]).optional(),
        limit: z.number().nullish(),
      }),
    },
    async (payload) => toContent(await handleExportConversations(payload, getUserId())),
  );

  // ============================================================================
  // Data Management Tools
  // ============================================================================

  server.registerTool(
    "data_status",
    {
      title: "Data Status",
      description: "Check presence of source files (conversations.json, memories.json) and generated data.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleDataStatus(getUserId())),
  );

  server.registerTool(
    "data_upload_conversations",
    {
      title: "Upload Conversations",
      description: "Upload conversations.json file. Overwrites existing file if present.",
      inputSchema: z.object({
        data: z.union([z.string(), z.any()]).describe("JSON content as string or object"),
      }),
    },
    async ({ data }) => toContent(await handleDataUploadConversations({ data }, getUserId())),
  );

  server.registerTool(
    "data_upload_memories",
    {
      title: "Upload Memories",
      description: "Upload memories.json file. Overwrites existing file if present.",
      inputSchema: z.object({
        data: z.union([z.string(), z.any()]).describe("JSON content as string or object"),
      }),
    },
    async ({ data }) => toContent(await handleDataUploadMemories({ data }, getUserId())),
  );

  server.registerTool(
    "data_clean",
    {
      title: "Clean Directory",
      description: "Clean generated data from a directory. Keeps source files (conversations.json, memories.json). Allowed: conversations, memory, models, training_data, adapters.",
      inputSchema: z.object({
        directory: z.enum(["conversations", "memory", "models_identity", "models_eeg", "training_data", "adapters"]),
      }),
    },
    async ({ directory }) => toContent(await handleDataClean({ directory }, getUserId())),
  );

  server.registerTool(
    "data_delete_source",
    {
      title: "Delete Source File",
      description: "Delete a source file (conversations.json or memories.json). Use this to remove uploaded source files.",
      inputSchema: z.object({
        type: z.enum(["conversations", "memories"]),
      }),
    },
    async ({ type }) => toContent(await handleDataDeleteSource({ type }, getUserId())),
  );

  server.registerTool(
    "data_conversations_list",
    {
      title: "List Conversations",
      description: "List all parsed conversation files with metadata (ID, title, message count, date range).",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleDataConversationsList(getUserId())),
  );

  server.registerTool(
    "data_conversation_get",
    {
      title: "Get Conversation",
      description: "Get specific conversation file content by ID.",
      inputSchema: z.object({
        id: z.string().describe("Conversation ID"),
      }),
    },
    async ({ id }) => toContent(await handleDataConversationGet({ id }, getUserId())),
  );

  server.registerTool(
    "data_conversation_update",
    {
      title: "Update Conversation",
      description: "Update conversation file content by ID.",
      inputSchema: z.object({
        id: z.string().describe("Conversation ID"),
        content: z.string().describe("New JSONL content"),
      }),
    },
    async ({ id, content }) => toContent(await handleDataConversationUpdate({ id, content }, getUserId())),
  );

  server.registerTool(
    "data_memories_list",
    {
      title: "List All Memory Records",
      description: "List all memory records from all JSONL files in memory directory.",
      inputSchema: z.object({}),
    },
    async () => toContent(await handleDataMemoriesList(getUserId())),
  );

  server.registerTool(
    "data_memory_file_get",
    {
      title: "Get Memory File",
      description: "Get specific memory JSONL file content by filename.",
      inputSchema: z.object({
        filename: z.string().describe("Memory filename (e.g., 'identity.jsonl')"),
      }),
    },
    async ({ filename }) => toContent(await handleDataMemoryFileGet({ filename }, getUserId())),
  );

  server.registerTool(
    "data_memory_file_update",
    {
      title: "Update Memory File",
      description: "Update memory JSONL file content by filename.",
      inputSchema: z.object({
        filename: z.string().describe("Memory filename (e.g., 'identity.jsonl')"),
        content: z.string().describe("New JSONL content"),
      }),
    },
    async ({ filename, content }) => toContent(await handleDataMemoryFileUpdate({ filename, content }, getUserId())),
  );
}

mcpProtocolRouter.post("/", async (req: Request, res: Response) => {
  const requestId = req.body?.id ?? "unknown";
  const sessionId = req.headers["mcp-session-id"] as string | undefined;
  
  // Log initialize request metadata for troubleshooting
  if (isInitializeRequest(req.body)) {
    logger.info("MCP Protocol: Initialize request", {
      hasClientInfo: !!req.body.params?.clientInfo,
      protocolVersion: req.body.params?.protocolVersion,
    });
  }

  logger.info("MCP Protocol: POST request", {
    hasSessionId: !!sessionId,
    sessionId: sessionId ? sessionId.substring(0, 8) + "..." : undefined,
    requestId,
  });

  try {
    let transport: StreamableHTTPServerTransport | undefined;

    // Extract user context from request
    const userContext = getUserContext(req);
    const userId = userContext.userId;

    // 1) Existing session: reuse its transport
    if (sessionId && transports[sessionId]) {
      const storedUserId = sessionUsers[sessionId];
      // Validate userId matches stored session (prevent cross-user access)
      if (userId && storedUserId && userId !== storedUserId) {
        logger.warn("MCP Protocol: User mismatch for session, invalidating", {
          sessionId: sessionId.substring(0, 8) + "...",
          storedUserId: storedUserId.substring(0, 8) + "...",
          newUserId: userId.substring(0, 8) + "...",
        });
        // Invalidate session - user changed
        delete transports[sessionId];
        delete sessionUsers[sessionId];
        // Fall through to create new session
      } else {
        transport = transports[sessionId];
        // Update user context if provided and matches
        if (userId) {
          sessionUsers[sessionId] = userId;
        }
        logger.info("MCP Protocol: Using existing session transport", {
          sessionId: sessionId.substring(0, 8) + "...",
          userId: userId ? userId.substring(0, 8) + "..." : "anonymous",
        });
      }
    }
    
    // Create new session if transport wasn't set (either new request or invalidated session)
    if (!transport && isInitializeRequest(req.body)) {
      // New initialization request: create fresh transport + server
      const newTransport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (initializedSessionId) => {
          logger.info("MCP Protocol: Session initialized", {
            sessionId: initializedSessionId.substring(0, 8) + "...",
            userId: userId ? userId.substring(0, 8) + "..." : "anonymous",
          });
          transports[initializedSessionId] = newTransport;
          // Store user context for this session
          sessionUsers[initializedSessionId] = userId;
        },
      });

      const server = new McpServer(
        {
          name: "identity-mcp",
          version: "0.1.0",
        },
        { capabilities: { tools: { listChanged: true } } },
      );

      // Create userId provider function for this session
      const getUserId = () => {
        const sessionId = newTransport.sessionId;
        return sessionId ? (sessionUsers[sessionId] || null) : userId;
      };

      registerTools(server, getUserId);
      await server.connect(newTransport);

      // Ensure cleanup when the transport closes
      newTransport.onclose = () => {
        const id = newTransport.sessionId;
        if (id && transports[id]) {
          logger.warn("MCP Protocol: Session transport closed", {
            sessionId: id.substring(0, 8) + "...",
            note: "Session state is preserved. The session can be resumed automatically on reconnection.",
          });
          // Note: We don't delete the transport immediately - let it clean up naturally
          delete transports[id];
          delete sessionUsers[id];
        }
      };

      transport = newTransport;
      logger.info("MCP Protocol: Created new session transport");
    } else {
      // 3) Invalid request (no session for non-initialize)
      logger.warn(
        "MCP Protocol: POST without valid session ID or initialize payload",
      );
      res.status(400).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid session ID provided",
        },
        id: null,
      });
      return;
    }

    // Delegate full JSON-RPC handling to the MCP SDK transport
    if (!transport) {
      res.status(400).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Bad Request: No valid transport available",
        },
        id: null,
      });
      return;
    }
    
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    logger.error("MCP Protocol: Error handling POST request", {
      error: String(error),
    });
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Internal server error",
        },
        id: requestId,
      });
    }
  }
});

mcpProtocolRouter.get("/", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;
  
  logger.info("MCP Protocol: GET request (SSE stream)", {
    hasSessionId: !!sessionId,
    sessionId: sessionId?.substring(0, 8) + "...",
  });

  try {
    if (!sessionId || !transports[sessionId]) {
      logger.warn("MCP Protocol: GET with invalid or missing session ID", {
        sessionId: sessionId?.substring(0, 8) + "...",
      });
      res.status(400).send("Invalid or missing session ID");
      return;
    }

    const transport = transports[sessionId];

    logger.info("MCP Protocol: Handling GET request for SSE stream", {
      sessionId: sessionId.substring(0, 8) + "...",
    });

    // Set extended timeout and keepalive for long-running sessions
    res.setTimeout(300000); // 5 minutes
    res.setHeader("Keep-Alive", "timeout=300, max=1000");
    
    // Send periodic keepalive comments to prevent connection timeout
    const keepAliveInterval = setInterval(() => {
      if (!res.headersSent) return;
      try {
        res.write(": keepalive\n\n");
      } catch (error) {
        clearInterval(keepAliveInterval);
      }
    }, 30000); // Every 30 seconds
    
    // Clean up interval when connection closes
    req.on("close", () => {
      clearInterval(keepAliveInterval);
    });
    
    await transport.handleRequest(req, res);
    
    clearInterval(keepAliveInterval);

    logger.info("MCP Protocol: SSE stream request completed", {
      sessionId: sessionId.substring(0, 8) + "...",
      headersSent: res.headersSent,
    });
  } catch (error) {
    logger.error("MCP Protocol: Unexpected error in GET handler", { 
      error: String(error),
      errorStack: error instanceof Error ? error.stack : undefined
    });
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

mcpProtocolRouter.delete("/", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;
  
  logger.info("MCP Protocol: DELETE request", { 
    hasSessionId: !!sessionId,
    sessionId: sessionId?.substring(0, 8) + "..." 
  });

  try {
    if (!sessionId || !transports[sessionId]) {
      logger.warn("MCP Protocol: DELETE with invalid or missing session ID", {
        sessionId: sessionId?.substring(0, 8) + "...",
      });
      res.status(400).send("Invalid or missing session ID");
      return;
    }

    const transport = transports[sessionId];

    await transport.handleRequest(req, res);

    logger.info("MCP Protocol: Session delete handled by transport", {
      sessionId: sessionId.substring(0, 8) + "...",
    });
  } catch (error) {
    logger.error("MCP Protocol: Error handling DELETE request", { 
      error: String(error),
      sessionId: sessionId ? sessionId.substring(0, 8) + "..." : undefined 
    });
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: "Internal server error",
        },
        id: null,
      });
    }
  }
});

