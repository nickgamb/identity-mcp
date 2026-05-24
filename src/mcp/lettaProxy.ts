/**
 * Letta REST API proxy — provides typed access to the Letta agent server.
 * All calls resolve the agent by name, cache the ID, and fall back
 * gracefully when Letta is unreachable.
 */

import { config } from "../config";
import { logger } from "../utils/logger";

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
function resolveSleeptimeFrequency(agentData: any): number {
  const group = agentData?.multi_agent_group ?? agentData?.managed_group;
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
  total?: number;
  error?: string;
}

export interface LettaMessage {
  id: string;
  role: string;
  content: string | null;
  created_at?: string;
  tool_calls?: Array<{
    name: string;
    arguments: string;
  }>;
  tool_call_id?: string;
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
 * @param sort  "oldest" (default) pages forward with `after`;
 *              "newest" pages backward with `before`.
 */
export async function getLettaArchival(
  limit: number = 50,
  cursor?: string,
  sort: "oldest" | "newest" = "oldest"
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
    let url = `/v1/agents/${agentId}/archival-memory?limit=${limit}`;
    if (sort === "newest") url += `&reverse=true`;
    if (cursor) {
      url +=
        sort === "newest"
          ? `&before=${encodeURIComponent(cursor)}`
          : `&after=${encodeURIComponent(cursor)}`;
    }

    const data = await lettaFetch(url);
    const passages = (Array.isArray(data) ? data : data.passages || []).map(
      (p: any) => ({
        id: p.id || "",
        text: p.content || p.text || "",
        created_at: p.created_at || p.timestamp,
        metadata: p.metadata || {},
      })
    );

    return {
      available: true,
      passages,
      total: data.total,
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
    let url = `/v1/agents/${agentId}/messages?limit=${limit}`;
    if (cursor) url += `&after=${encodeURIComponent(cursor)}`;

    const data = await lettaFetch(url);
    const rawMessages = Array.isArray(data) ? data : data.messages || [];

    const messages: LettaMessage[] = rawMessages.map((m: any) => ({
      id: m.id || "",
      role: m.role || m.message_type || "unknown",
      content:
        typeof m.content === "string"
          ? m.content
          : m.content?.text ||
            m.content?.content ||
            (m.content ? JSON.stringify(m.content) : null),
      created_at: m.created_at || m.timestamp,
      tool_calls: m.tool_calls?.map((tc: any) => ({
        name: tc.function?.name || tc.name || "unknown",
        arguments:
          typeof tc.function?.arguments === "string"
            ? tc.function.arguments
            : JSON.stringify(tc.function?.arguments || tc.arguments || {}),
      })),
      tool_call_id: m.tool_call_id,
    }));

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

  const body: Record<string, unknown> = {};

  if (patch.enable_sleeptime !== undefined) {
    body.enable_sleeptime = patch.enable_sleeptime;
  }
  if (patch.sleeptime_agent_frequency !== undefined) {
    body.sleeptime_agent_frequency = patch.sleeptime_agent_frequency;
  }
  if (patch.model !== undefined) {
    body.model = toOllamaHandle(String(patch.model));
  }
  if (patch.embedding !== undefined) {
    body.embedding = toOllamaHandle(String(patch.embedding));
  }
  if (patch.timezone !== undefined) {
    body.timezone = patch.timezone;
  }
  if (patch.description !== undefined) {
    body.description = patch.description;
  }

  if (Object.keys(body).length === 0) {
    return { success: false, error: "No supported config fields in patch" };
  }

  try {
    await lettaFetch(`/v1/agents/${agentId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    // Note: agent ID doesn't change from a config PATCH, so no need to clear cache
    return { success: true };
  } catch (e) {
    logger.error("updateLettaConfig failed", { error: String(e) });
    return { success: false, error: String(e) };
  }
}
