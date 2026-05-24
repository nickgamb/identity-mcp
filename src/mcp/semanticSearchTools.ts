/**
 * Semantic search via Letta archival memory (pgvector embeddings).
 * Falls back gracefully when Letta is unavailable.
 */

import { logger } from "../utils/logger";

const LETTA_BASE_URL = process.env.LETTA_BASE_URL || "http://letta:8283";
const LETTA_AGENT_NAME = process.env.LETTA_AGENT_NAME || "identity";

let cachedAgentId: string | null = null;

async function getAgentId(): Promise<string | null> {
  if (cachedAgentId) return cachedAgentId;
  try {
    const resp = await fetch(`${LETTA_BASE_URL}/v1/agents?name=${LETTA_AGENT_NAME}`);
    if (!resp.ok) return null;
    const agents = await resp.json() as any[];
    const agent = agents.find((a: any) => a.name === LETTA_AGENT_NAME);
    if (agent) {
      cachedAgentId = agent.id;
      return agent.id;
    }
  } catch (e) {
    logger.warn("Could not reach Letta to resolve agent ID", { error: String(e) });
  }
  return null;
}

export interface SemanticSearchRequest {
  query: string;
  limit?: number;
}

export interface SemanticSearchResult {
  text: string;
  created_at?: string;
  score?: number;
  id?: string;
}

export interface SemanticSearchResponse {
  results: SemanticSearchResult[];
  count: number;
  source: "letta_archival";
  agent_id?: string;
  error?: string;
}

export async function handleSemanticSearch(
  req: SemanticSearchRequest,
  _userId: string | null = null
): Promise<SemanticSearchResponse> {
  const agentId = await getAgentId();
  if (!agentId) {
    return {
      results: [],
      count: 0,
      source: "letta_archival",
      error: "Letta agent not available. Ensure Letta is running and the identity agent exists.",
    };
  }

  try {
    const topK = req.limit ?? 20;
    const url = `${LETTA_BASE_URL}/v1/agents/${agentId}/archival-memory/search?query=${encodeURIComponent(req.query)}&top_k=${topK}`;
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json" },
    });

    if (!resp.ok) {
      const detail = await resp.text().catch(() => "");
      return {
        results: [],
        count: 0,
        source: "letta_archival",
        agent_id: agentId,
        error: `Letta returned ${resp.status}: ${detail.slice(0, 200)}`,
      };
    }

    const body = await resp.json() as any;
    const passages = body.results || body || [];
    const results: SemanticSearchResult[] = passages.map((p: any) => ({
      text: p.content || p.text || "",
      created_at: p.timestamp || p.created_at,
      score: p.score,
      id: p.id,
    }));

    return {
      results,
      count: results.length,
      source: "letta_archival",
      agent_id: agentId,
    };
  } catch (e) {
    logger.error("Semantic search failed", { error: String(e) });
    return {
      results: [],
      count: 0,
      source: "letta_archival",
      agent_id: agentId,
      error: String(e),
    };
  }
}
