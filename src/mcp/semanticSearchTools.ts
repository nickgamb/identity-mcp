/**
 * Semantic search via Letta archival memory (pgvector embeddings).
 * Falls back gracefully when Letta is unavailable.
 */

import { config } from "../config";
import { logger } from "../utils/logger";
import { getAgentId } from "./lettaProxy";

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
    const targetLimit = req.limit ?? 20;
    // Files are densely topical and tend to crowd out conversations in a
    // single top-K (we've seen 100/100 files for some queries). Issue
    // separate tag-filtered searches per type so every type gets a fair
    // shot, then interleave by Letta's per-bucket rank.
    //
    // Letta's `tags` is a post-filter applied to the vector top-K, not a
    // pre-filter — so we have to over-fetch per bucket to ensure enough
    // survivors of the chosen type. Empirically, ~3x the target works.
    const perBucket = Math.max(targetLimit * 3, 30);
    const bucketResults = await Promise.all(
      INGEST_TAGS.map((tag) => searchByTag(agentId, req.query, tag, perBucket))
    );

    // Letta's per-tag responses come back already sorted by similarity within
    // their bucket; interleaving keeps that ranking while guaranteeing variety.
    const out: SemanticSearchResult[] = [];
    const seenIds = new Set<string>();
    let idx = 0;
    while (out.length < targetLimit) {
      let added = false;
      for (const bucket of bucketResults) {
        if (out.length >= targetLimit) break;
        if (idx < bucket.length) {
          const r = bucket[idx];
          if (r.id && seenIds.has(r.id)) continue;
          if (r.id) seenIds.add(r.id);
          out.push(r);
          added = true;
        }
      }
      if (!added) break;
      idx++;
    }

    return {
      results: out,
      count: out.length,
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

// Tags applied by letta/ingest.py to each passage type. Order matters:
// round-robin pulls in this priority so the agent always sees conversations
// alongside files even when files dominate by vector similarity.
const INGEST_TAGS = [
  "conversation",
  "file",
  "memory",
  "chatgpt_memory",
  "claude_memory",
];

async function searchByTag(
  agentId: string,
  query: string,
  tag: string,
  topK: number
): Promise<SemanticSearchResult[]> {
  const url =
    `${config.LETTA_BASE_URL}/v1/agents/${agentId}/archival-memory/search` +
    `?query=${encodeURIComponent(query)}&top_k=${topK}` +
    `&tags=${encodeURIComponent(tag)}&tag_match_mode=any`;
  try {
    const resp = await fetch(url, { headers: { "Content-Type": "application/json" } });
    if (!resp.ok) {
      logger.warn("Tag-filtered archival search failed", {
        tag,
        status: resp.status,
      });
      return [];
    }
    const body = await resp.json() as any;
    const passages = body.results || body || [];
    return passages.map((p: any) => ({
      text: p.content || p.text || "",
      created_at: p.timestamp || p.created_at,
      score: p.score,
      id: p.id,
    }));
  } catch (e) {
    logger.warn("Tag-filtered archival search error", { tag, error: String(e) });
    return [];
  }
}
