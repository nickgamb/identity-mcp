/**
 * Letta REST API proxy — provides typed access to the Letta agent server.
 * All calls resolve the agent by name, cache the ID, and fall back
 * gracefully when Letta is unreachable.
 */

import { config } from "../config";
import { logger } from "../utils/logger";
import {
  type ArchivalPassageType,
  matchesArchivalTypeFilter,
  passageDateKey,
  passageMatchesDateRange,
  preferNewestScanForType,
} from "../utils/archivalPassage";

export type ArchivalTypeFilter = ArchivalPassageType;

function toOllamaHandle(nameOrHandle: string): string {
  const trimmed = nameOrHandle.trim();
  if (!trimmed) return trimmed;
  return trimmed.startsWith("ollama/") ? trimmed : `ollama/${trimmed}`;
}

// ── Shared agent resolver (also used by semanticSearchTools) ────────────

let cachedAgentId: string | null = null;

export async function getAgentId(): Promise<string | null> {
  if (cachedAgentId) return cachedAgentId;
  try {
    const resp = await fetch(
      `${config.LETTA_BASE_URL}/v1/agents?name=${config.LETTA_AGENT_NAME}`
    );
    if (!resp.ok) return null;
    const agents = (await resp.json()) as any[];
    const agent = agents.find((a: any) => a.name === config.LETTA_AGENT_NAME);
    if (agent) {
      cachedAgentId = agent.id;
      return agent.id;
    }
  } catch (e) {
    logger.warn("Could not reach Letta to resolve agent ID", {
      error: String(e),
    });
  }
  return null;
}

/** Reset cached agent ID (useful after agent recreation). */
export function clearAgentIdCache(): void {
  cachedAgentId = null;
}

// ── Helper ──────────────────────────────────────────────────────────────

async function lettaFetch(path: string, init?: RequestInit): Promise<any> {
  const resp = await fetch(`${config.LETTA_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => "");
    throw new Error(`Letta ${resp.status}: ${detail.slice(0, 300)}`);
  }
  return resp.json();
}

/** Letta 0.16+ stores sleeptime frequency on the agent group, not top-level. */
function resolveManagedGroup(agentData: any): any | null {
  return agentData?.multi_agent_group ?? agentData?.managed_group ?? null;
}

function resolveSleeptimeFrequency(agentData: any): number {
  const group = resolveManagedGroup(agentData);
  if (group?.sleeptime_agent_frequency != null) {
    return group.sleeptime_agent_frequency;
  }
  return agentData?.sleeptime_agent_frequency ?? 0;
}

function resolveRawMemoryBlocks(agentData: any, memoryData: any): any[] {
  return (
    memoryData?.blocks ||
    memoryData?.memory?.blocks ||
    agentData?.memory?.blocks ||
    agentData?.blocks ||
    []
  );
}

const ARCHIVAL_COUNT_TTL_MS = 10 * 60 * 1000;
let archivalCountCache: { agentId: string; count: number; at: number } | null =
  null;
let archivalCountRefresh: Promise<void> | null = null;

/** Full pagination is expensive (~20k passages); never block status on it. */
async function countArchivalPassages(agentId: string): Promise<number> {
  const pageSize = 1000;
  let total = 0;
  let after: string | undefined;

  for (let page = 0; page < 50; page++) {
    let url = `/v1/agents/${agentId}/archival-memory?limit=${pageSize}`;
    if (after) url += `&after=${encodeURIComponent(after)}`;

    const data = await lettaFetch(url);
    const items: any[] = Array.isArray(data) ? data : data.passages || [];
    if (items.length === 0) break;

    total += items.length;
    if (items.length < pageSize) break;

    const nextAfter = items[items.length - 1]?.id;
    if (!nextAfter || nextAfter === after) break;
    after = nextAfter;
  }

  return total;
}

function getCachedArchivalCount(agentId: string): number | undefined {
  if (
    archivalCountCache?.agentId === agentId &&
    Date.now() - archivalCountCache.at < ARCHIVAL_COUNT_TTL_MS
  ) {
    return archivalCountCache.count;
  }
  return undefined;
}

function refreshArchivalCountInBackground(agentId: string): void {
  if (archivalCountRefresh) return;
  archivalCountRefresh = countArchivalPassages(agentId)
    .then((count) => {
      archivalCountCache = { agentId, count, at: Date.now() };
    })
    .catch((e) => {
      logger.warn("Background archival count refresh failed", {
        error: String(e),
      });
    })
    .finally(() => {
      archivalCountRefresh = null;
    });
}

// ── Types ───────────────────────────────────────────────────────────────

export interface LettaMemoryBlock {
  id: string;
  label: string;
  value: string;
  limit: number;
  created_at?: string;
  updated_at?: string;
}

export interface LettaStatus {
  available: boolean;
  agent?: {
    id: string;
    name: string;
    model: string;
    embedding_model: string;
    /** Full Letta handle, e.g. ollama/qwen3:32b */
    model_handle: string;
    embedding_handle: string;
    timezone?: string;
    description?: string | null;
    created_at: string;
    enable_sleeptime?: boolean;
    sleeptime_agent_frequency: number;
    tool_count: number;
    tools: string[];
  };
  memory?: {
    blocks: Array<{
      label: string;
      char_count: number;
      limit: number;
    }>;
  };
  archival_count?: number;
  /** True while a background full count is running (cache cold). */
  archival_count_loading?: boolean;
  error?: string;
}

export interface LettaCoreMemory {
  available: boolean;
  blocks: LettaMemoryBlock[];
  error?: string;
}

export interface LettaArchivalPage {
  available: boolean;
  passages: Array<{
    id: string;
    text: string;
    created_at?: string;
    metadata?: Record<string, any>;
  }>;
  /** Cursor for the next page — when type-filtering, this is the scan
   *  position (not the last returned passage). */
  nextCursor?: string;
  /** False when Letta has no further pages to scan. */
  hasMore?: boolean;
  total?: number;
  error?: string;
}

function sortPassagesByDate<T extends { text: string; created_at?: string }>(
  items: T[],
  sort: "oldest" | "newest"
): T[] {
  return [...items].sort((a, b) => {
    const da = passageDateKey(a.text, a.created_at) ?? "";
    const db = passageDateKey(b.text, b.created_at) ?? "";
    if (!da && !db) return 0;
    if (!da) return 1;
    if (!db) return -1;
    return sort === "newest" ? db.localeCompare(da) : da.localeCompare(db);
  });
}

export interface LettaMessage {
  id: string;
  role: string;
  message_type: string;
  content: string | null;
  created_at?: string;
  tool_calls?: Array<{
    name: string;
    arguments: string;
    tool_call_id?: string;
  }>;
  tool_call_id?: string;
  tool_status?: "success" | "error";
  reasoning?: string;
  summary?: string;
  event_type?: string;
  name?: string;
  sender_id?: string;
  step_id?: string;
  run_id?: string;
  is_err?: boolean;
  approve?: boolean;
  approval_reason?: string;
}

function extractLettaTextContent(content: unknown): string | null {
  if (content == null) return null;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts = content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object") {
          const p = part as Record<string, unknown>;
          if (typeof p.text === "string") return p.text;
          if (typeof p.tool_return === "string") return p.tool_return;
        }
        return null;
      })
      .filter((p): p is string => !!p);
    return parts.length ? parts.join("\n") : JSON.stringify(content);
  }
  if (typeof content === "object") {
    const c = content as Record<string, unknown>;
    if (typeof c.text === "string") return c.text;
    if (typeof c.content === "string") return c.content;
    return JSON.stringify(content);
  }
  return String(content);
}

function mapToolCall(tc: any): {
  name: string;
  arguments: string;
  tool_call_id?: string;
} {
  const args = tc?.function?.arguments ?? tc?.arguments ?? {};
  return {
    name: tc?.function?.name || tc?.name || "unknown",
    arguments:
      typeof args === "string" ? args : JSON.stringify(args ?? {}, null, 2),
    tool_call_id: tc?.tool_call_id || tc?.id,
  };
}

function mapLettaMessage(m: any): LettaMessage {
  const messageType = m.message_type || m.role || "unknown";
  let role = "unknown";
  let content: string | null = null;
  let tool_calls: LettaMessage["tool_calls"];
  let tool_call_id = m.tool_call_id;
  let tool_status: LettaMessage["tool_status"];
  let reasoning: string | undefined;
  let summary: string | undefined;
  let event_type: string | undefined;
  let approve: boolean | undefined;
  let approval_reason: string | undefined;

  switch (messageType) {
    case "user_message":
      role = "user";
      content = extractLettaTextContent(m.content);
      break;
    case "assistant_message":
      role = "assistant";
      content = extractLettaTextContent(m.content ?? m.assistant_message);
      break;
    case "reasoning_message":
      role = "reasoning";
      reasoning =
        m.reasoning ||
        m.internal_monologue ||
        extractLettaTextContent(m.content) ||
        undefined;
      content = reasoning ?? null;
      break;
    case "hidden_reasoning_message":
      role = "reasoning";
      content = m.hidden_reasoning
        ? `[${m.state || "hidden"}] ${m.hidden_reasoning}`
        : `[${m.state || "hidden"} reasoning omitted]`;
      break;
    case "tool_call_message": {
      role = "tool_call";
      const calls = m.tool_calls?.length
        ? m.tool_calls
        : m.tool_call
          ? [m.tool_call]
          : [];
      tool_calls = calls.map(mapToolCall);
      if (tool_calls[0]) {
        tool_call_id = tool_calls[0].tool_call_id;
        content = `Calling ${tool_calls[0].name}`;
      }
      break;
    }
    case "tool_return_message":
      role = "tool";
      tool_call_id =
        m.tool_call_id || m.tool_returns?.[0]?.tool_call_id || tool_call_id;
      tool_status = m.status || m.tool_returns?.[0]?.status;
      content =
        extractLettaTextContent(m.tool_return) ||
        extractLettaTextContent(m.tool_returns?.[0]?.tool_return) ||
        null;
      if (m.stdout?.length) {
        content = `${content || ""}\n[stdout]\n${m.stdout.join("\n")}`.trim();
      }
      if (m.stderr?.length) {
        content = `${content || ""}\n[stderr]\n${m.stderr.join("\n")}`.trim();
      }
      break;
    case "system_message":
      role = "system";
      content = extractLettaTextContent(m.content);
      break;
    case "summary_message":
      role = "summary";
      summary = m.summary;
      content = summary ?? null;
      break;
    case "event_message":
      role = "event";
      event_type = m.event_type;
      content = m.event_data
        ? JSON.stringify(m.event_data, null, 2)
        : m.event_type || null;
      break;
    case "approval_request_message": {
      role = "approval";
      const tc = m.tool_call || m.tool_calls?.[0];
      if (tc) {
        tool_calls = [mapToolCall(tc)];
        content = `Approval requested: ${tool_calls[0].name}`;
      }
      break;
    }
    case "approval_response_message":
      role = "approval";
      approve = m.approve;
      approval_reason = m.reason;
      content =
        approve === true
          ? "Tool execution approved"
          : approve === false
            ? "Tool execution denied"
            : "Approval response";
      break;
    default:
      role = messageType.replace(/_message$/, "") || m.role || "unknown";
      content =
        extractLettaTextContent(m.content) ||
        extractLettaTextContent(m.assistant_message) ||
        null;
      if (m.tool_calls?.length) {
        tool_calls = m.tool_calls.map(mapToolCall);
      }
      break;
  }

  if (!tool_calls && m.tool_calls?.length) {
    tool_calls = m.tool_calls.map(mapToolCall);
  }

  return {
    id: m.id || "",
    role,
    message_type: messageType,
    content,
    created_at: m.created_at || m.date,
    tool_calls,
    tool_call_id,
    tool_status,
    reasoning,
    summary,
    event_type,
    name: m.name,
    sender_id: m.sender_id,
    step_id: m.step_id,
    run_id: m.run_id,
    is_err: m.is_err,
    approve,
    approval_reason,
  };
}

export interface LettaMessagesPage {
  available: boolean;
  messages: LettaMessage[];
  error?: string;
}

// ── API functions ───────────────────────────────────────────────────────

/**
 * Get full agent status: metadata, memory summary, archival count.
 */
export async function getLettaStatus(): Promise<LettaStatus> {
  const agentId = await getAgentId();
  if (!agentId) {
    return { available: false, error: "Letta agent not available" };
  }

  try {
    // Fetch agent details and memory in parallel
    const [agentData, memoryData] = await Promise.all([
      lettaFetch(`/v1/agents/${agentId}`),
      lettaFetch(`/v1/agents/${agentId}/memory`).catch(() => null),
    ]);

    // Extract tool names
    const tools: string[] = (agentData.tools || []).map(
      (t: any) => t.name || t
    );

    // Extract memory blocks summary (GET /memory 404s on Letta 0.16 — use agent payload)
    const blocks = resolveRawMemoryBlocks(agentData, memoryData).map(
      (b: any) => ({
        label: b.label || b.name || "unknown",
        char_count: (b.value || "").length,
        limit: b.limit || 0,
      })
    );

    const archivalCount = getCachedArchivalCount(agentId);
    if (archivalCount === undefined) {
      refreshArchivalCountInBackground(agentId);
    }

    const sleeptimeFreq = resolveSleeptimeFrequency(agentData);

    return {
      available: true,
      agent: {
        id: agentId,
        name: agentData.name || config.LETTA_AGENT_NAME,
        model: agentData.llm_config?.model || agentData.model || "unknown",
        embedding_model:
          agentData.embedding_config?.embedding_model ||
          agentData.embedding_model ||
          "unknown",
        model_handle:
          agentData.llm_config?.handle ||
          agentData.model ||
          "unknown",
        embedding_handle:
          agentData.embedding_config?.handle ||
          agentData.embedding ||
          "unknown",
        timezone: agentData.timezone || "UTC",
        description: agentData.description ?? null,
        created_at: agentData.created_at || "",
        enable_sleeptime: agentData.enable_sleeptime === true,
        sleeptime_agent_frequency: sleeptimeFreq,
        tool_count: tools.length,
        tools,
      },
      memory: { blocks },
      archival_count: archivalCount,
      archival_count_loading:
        archivalCount === undefined && archivalCountRefresh !== null,
    };
  } catch (e) {
    logger.error("getLettaStatus failed", { error: String(e) });
    return { available: false, error: String(e) };
  }
}

/**
 * Get full core memory blocks (persona, human, etc.) with content.
 */
export async function getLettaCoreMemory(): Promise<LettaCoreMemory> {
  const agentId = await getAgentId();
  if (!agentId) {
    return { available: false, blocks: [], error: "Letta agent not available" };
  }

  try {
    const [memoryData, agentData] = await Promise.all([
      lettaFetch(`/v1/agents/${agentId}/memory`).catch(() => null),
      lettaFetch(`/v1/agents/${agentId}`),
    ]);
    const rawBlocks = resolveRawMemoryBlocks(agentData, memoryData);

    const blocks: LettaMemoryBlock[] = rawBlocks.map((b: any) => ({
      id: b.id || "",
      label: b.label || b.name || "unknown",
      value: b.value || "",
      limit: b.limit || 0,
      created_at: b.created_at,
      updated_at: b.updated_at,
    }));

    return { available: true, blocks };
  } catch (e) {
    logger.error("getLettaCoreMemory failed", { error: String(e) });
    return { available: false, blocks: [], error: String(e) };
  }
}

/**
 * Update a single core memory block's value.
 */
export async function updateLettaCoreMemory(
  blockLabel: string,
  value: string
): Promise<{ success: boolean; block?: LettaMemoryBlock; error?: string }> {
  const agentId = await getAgentId();
  if (!agentId) {
    return { success: false, error: "Letta agent not available" };
  }

  try {
    // Get blocks from both sources to handle Letta 0.16+ API shape differences
    const [memoryData, agentData] = await Promise.all([
      lettaFetch(`/v1/agents/${agentId}/memory`).catch(() => null),
      lettaFetch(`/v1/agents/${agentId}`),
    ]);
    const rawBlocks = resolveRawMemoryBlocks(agentData, memoryData);
    const block = rawBlocks.find(
      (b: any) => (b.label || b.name) === blockLabel
    );

    if (!block) {
      return { success: false, error: `Block "${blockLabel}" not found` };
    }

    // Update the block
    const updated = await lettaFetch(`/v1/blocks/${block.id}`, {
      method: "PATCH",
      body: JSON.stringify({ value }),
    });

    return {
      success: true,
      block: {
        id: updated.id || block.id,
        label: updated.label || blockLabel,
        value: updated.value || value,
        limit: updated.limit || block.limit || 0,
        updated_at: updated.updated_at,
      },
    };
  } catch (e) {
    logger.error("updateLettaCoreMemory failed", { error: String(e) });
    return { success: false, error: String(e) };
  }
}

/**
 * Get paginated archival memory passages.
 *
 * @param sort        "oldest" (default) pages forward with `after`;
 *                    "newest" pages backward with `before`.
 * @param typeFilter  Optional — when set, the proxy scans through Letta
 *                    in larger batches and returns only passages whose
 *                    text prefix matches the requested type.
 */
export async function getLettaArchival(
  limit: number = 50,
  cursor?: string,
  sort: "oldest" | "newest" = "oldest",
  typeFilter?: ArchivalTypeFilter,
  dateFrom?: string,
  dateTo?: string
): Promise<LettaArchivalPage> {
  const agentId = await getAgentId();
  if (!agentId) {
    return {
      available: false,
      passages: [],
      error: "Letta agent not available",
    };
  }

  try {
    const hasDateFilter = !!(dateFrom || dateTo);

    const mapPassage = (p: any) => ({
      id: p.id || "",
      text: p.content || p.text || "",
      created_at: p.created_at || p.timestamp,
      metadata: p.metadata || {},
    });

    const passageOk = (text: string, createdAt?: string) => {
      if (typeFilter && !matchesArchivalTypeFilter(text, typeFilter)) {
        return false;
      }
      if (hasDateFilter && !passageMatchesDateRange(text, createdAt, dateFrom, dateTo)) {
        return false;
      }
      return true;
    };

    // ── Unfiltered path (fast — single request) ──────────────────
    if (!typeFilter && !hasDateFilter) {
      let url = `/v1/agents/${agentId}/archival-memory?limit=${limit}`;
      if (sort === "newest") url += `&ascending=false`;
      if (cursor) {
        url +=
          sort === "newest"
            ? `&before=${encodeURIComponent(cursor)}`
            : `&after=${encodeURIComponent(cursor)}`;
      }

      const data = await lettaFetch(url);
      const raw = Array.isArray(data) ? data : data.passages || [];
      const passages = raw.map(mapPassage);

      const nextCursor =
        raw.length > 0 ? raw[raw.length - 1].id : undefined;

      return {
        available: true,
        passages,
        nextCursor,
        hasMore: raw.length >= limit && !!nextCursor,
        total: data.total,
      };
    }

    // ── Filtered path — scan & filter server-side ────────────────
    // file/memory/analysis ingest after conversations — scan from newest unless
    // filtering conversations only.
    const scanNewest = preferNewestScanForType(typeFilter);
    const scanSort: "oldest" | "newest" = scanNewest ? "newest" : sort;
    const BATCH_SIZE = 500;
    const MAX_SCAN = 25_000;
    let scanned = 0;
    let scanCursor = cursor;
    let exhausted = false;
    const matched: any[] = [];

    while (matched.length < limit && scanned < MAX_SCAN) {
      let url = `/v1/agents/${agentId}/archival-memory?limit=${BATCH_SIZE}`;
      if (scanSort === "newest") url += `&ascending=false`;
      if (scanCursor) {
        url +=
          scanSort === "newest"
            ? `&before=${encodeURIComponent(scanCursor)}`
            : `&after=${encodeURIComponent(scanCursor)}`;
      }

      const data = await lettaFetch(url);
      const batch = Array.isArray(data) ? data : data.passages || [];
      if (batch.length === 0) break;

      scanned += batch.length;
      scanCursor = batch[batch.length - 1].id;

      for (const p of batch) {
        const text = p.content || p.text || "";
        const createdAt = p.created_at || p.timestamp;
        if (passageOk(text, createdAt)) {
          matched.push(p);
          if (matched.length >= limit) break;
        }
      }

      if (batch.length < BATCH_SIZE) {
        exhausted = true;
        break;
      }
    }

    let passages = sortPassagesByDate(
      matched.map(mapPassage),
      sort
    );

    return {
      available: true,
      passages,
      nextCursor: scanCursor,
      hasMore: !exhausted && !!scanCursor,
    };
  } catch (e) {
    logger.error("getLettaArchival failed", { error: String(e) });
    return { available: false, passages: [], error: String(e) };
  }
}

/**
 * Get recent agent messages (conversation + sleeptime activity).
 */
export async function getLettaMessages(
  limit: number = 100,
  cursor?: string
): Promise<LettaMessagesPage> {
  const agentId = await getAgentId();
  if (!agentId) {
    return {
      available: false,
      messages: [],
      error: "Letta agent not available",
    };
  }

  try {
    let url = `/v1/agents/${agentId}/messages?limit=${limit}&order=desc&include_err=true`;
    if (cursor) url += `&after=${encodeURIComponent(cursor)}`;

    const data = await lettaFetch(url);
    const rawMessages = Array.isArray(data) ? data : data.messages || [];

    const messages: LettaMessage[] = rawMessages.map(mapLettaMessage);

    return { available: true, messages };
  } catch (e) {
    logger.error("getLettaMessages failed", { error: String(e) });
    return { available: false, messages: [], error: String(e) };
  }
}

/**
 * List models available in the local Ollama instance (for Letta model pickers).
 */
export async function listOllamaModels(): Promise<{
  available: boolean;
  models: string[];
  error?: string;
}> {
  try {
    const resp = await fetch(`${config.OLLAMA_BASE_URL}/api/tags`);
    if (!resp.ok) {
      throw new Error(`Ollama ${resp.status}`);
    }
    const data = (await resp.json()) as { models?: Array<{ name: string }> };
    const models = (data.models || [])
      .map((m) => m.name)
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b));
    return { available: true, models };
  } catch (e) {
    logger.warn("listOllamaModels failed", { error: String(e) });
    return { available: false, models: [], error: String(e) };
  }
}

/**
 * Update agent configuration (sleeptime, models, timezone, etc.).
 */
export async function updateLettaConfig(
  patch: Record<string, any>
): Promise<{ success: boolean; error?: string }> {
  const agentId = await getAgentId();
  if (!agentId) {
    return { success: false, error: "Letta agent not available" };
  }

  const hasAgentFields =
    patch.enable_sleeptime !== undefined ||
    patch.model !== undefined ||
    patch.embedding !== undefined ||
    patch.timezone !== undefined ||
    patch.description !== undefined ||
    (patch.sleeptime_agent_frequency !== undefined &&
      patch.sleeptime_agent_frequency !== null);

  if (!hasAgentFields) {
    return { success: false, error: "No supported config fields in patch" };
  }

  try {
    const agentData = await lettaFetch(`/v1/agents/${agentId}`);
    const group = resolveManagedGroup(agentData);
    const groupId = group?.id as string | undefined;

    const agentBody: Record<string, unknown> = {};
    if (patch.enable_sleeptime !== undefined) {
      agentBody.enable_sleeptime = patch.enable_sleeptime;
    }
    if (patch.model !== undefined) {
      agentBody.model = toOllamaHandle(String(patch.model));
    }
    if (patch.embedding !== undefined) {
      agentBody.embedding = toOllamaHandle(String(patch.embedding));
    }
    if (patch.timezone !== undefined) {
      agentBody.timezone = patch.timezone;
    }
    if (patch.description !== undefined) {
      agentBody.description = patch.description;
    }
    // Legacy: frequency on agent only when there is no sleeptime group
    if (
      patch.sleeptime_agent_frequency !== undefined &&
      !groupId
    ) {
      agentBody.sleeptime_agent_frequency = patch.sleeptime_agent_frequency;
    }

    if (Object.keys(agentBody).length > 0) {
      await lettaFetch(`/v1/agents/${agentId}`, {
        method: "PATCH",
        body: JSON.stringify(agentBody),
      });
    }

    if (
      groupId &&
      patch.sleeptime_agent_frequency !== undefined &&
      patch.enable_sleeptime !== false
    ) {
      const freq = Number(patch.sleeptime_agent_frequency);
      await lettaFetch(`/v1/groups/${groupId}`, {
        method: "PATCH",
        body: JSON.stringify({
          manager_config: {
            manager_type: group?.manager_type ?? "sleeptime",
            sleeptime_agent_frequency: freq,
          },
        }),
      });
      logger.info("Updated sleeptime group frequency", { groupId, freq });
    }

    return { success: true };
  } catch (e) {
    logger.error("updateLettaConfig failed", { error: String(e) });
    return { success: false, error: String(e) };
  }
}
