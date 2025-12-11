import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";

interface IdentityRecord {
  type: string;
  data: unknown;
}

interface MomentumData {
  early_avg: number;
  late_avg: number;
  change_percent: number;
  trend: "rising" | "falling" | "stable";
}

interface NamingEvent {
  conversation_id: string;
  timestamp: string;
  role: string;
  name: string;
  context: string;
}

let cachedAnalysis: IdentityRecord[] | null = null;

function loadIdentityAnalysis(): IdentityRecord[] {
  if (cachedAnalysis) {
    return cachedAnalysis;
  }

  const filePath = path.join(config.MEMORY_DIR, "identity_analysis.jsonl");

  if (!fs.existsSync(filePath)) {
    logger.warn("Identity analysis file not found", { path: filePath });
    return [];
  }

  try {
    const content = fs.readFileSync(filePath, "utf8");
    const lines = content.split("\n").filter((line) => line.trim());
    cachedAnalysis = lines.map((line) => JSON.parse(line));
    logger.info("Loaded identity analysis", { records: cachedAnalysis.length });
    return cachedAnalysis;
  } catch (error) {
    logger.error("Failed to load identity analysis", { error: String(error) });
    return [];
  }
}

function getRecordByType(type: string): unknown | null {
  const records = loadIdentityAnalysis();
  const record = records.find((r) => r.type === type);
  return record?.data || null;
}

/**
 * Get identity analysis summary
 */
export async function handleIdentityAnalysisSummary() {
  const summary = getRecordByType("identity.summary");
  // New structure: human_identity and relational_context
  const humanIdentity = getRecordByType("identity.human") as Record<string, unknown> | null;
  const relationalContext = getRecordByType("identity.relational_context") as Record<string, unknown> | null;
  
  // Backward compatibility: try old structure if new one not found
  const relational = getRecordByType("identity.relational") as Record<string, unknown> | null;
  const selfRef = getRecordByType("identity.self_referential") as Record<string, unknown> | null;

  if (!summary) {
    return {
      available: false,
      message: "Identity analysis not found. Run analyze_identity.py first.",
    };
  }

  // Use new structure if available, otherwise fall back to old structure
  if (humanIdentity) {
    return {
      available: true,
      summary,
      human_identity_overview: {
        we_i_ratio: humanIdentity.we_vs_i_ratio as number,
        relational_patterns: Object.keys((humanIdentity.relational_patterns as Record<string, unknown>) || {}),
        self_referential_patterns: Object.keys((humanIdentity.self_referential_patterns as Record<string, unknown>) || {}),
      },
      relational_context: relationalContext
        ? {
            we_i_ratio_assistant: relationalContext.we_vs_i_ratio_assistant as number,
            note: relationalContext.note as string,
          }
        : null,
    };
  }

  // Backward compatibility with old structure
  return {
    available: true,
    summary,
    relational_overview: relational
      ? {
          we_i_ratio_user: relational.we_vs_i_ratio_user,
          we_i_ratio_assistant: relational.we_vs_i_ratio_assistant,
        }
      : null,
    self_referential_overview: selfRef
      ? {
          user_patterns: Object.keys((selfRef.user_patterns as Record<string, unknown>) || {}),
          assistant_patterns: Object.keys((selfRef.assistant_patterns as Record<string, unknown>) || {}),
        }
      : null,
  };
}

/**
 * Get pattern momentum - what's rising/falling over time
 */
export async function handleIdentityGetMomentum() {
  const momentum = getRecordByType("identity.momentum") as Record<string, Record<string, MomentumData>> | null;

  if (!momentum) {
    return {
      available: false,
      message: "Momentum data not found. Run analyze_identity.py first.",
    };
  }

  // Extract rising and falling patterns
  const rising: { pattern: string; change: number; category: string }[] = [];
  const falling: { pattern: string; change: number; category: string }[] = [];
  const stable: { pattern: string; category: string }[] = [];

  for (const [category, patterns] of Object.entries(momentum)) {
    for (const [pattern, data] of Object.entries(patterns)) {
      if (data.trend === "rising") {
        rising.push({ pattern, change: data.change_percent, category });
      } else if (data.trend === "falling") {
        falling.push({ pattern, change: data.change_percent, category });
      } else {
        stable.push({ pattern, category });
      }
    }
  }

  // Sort by magnitude of change
  rising.sort((a, b) => b.change - a.change);
  falling.sort((a, b) => a.change - b.change);

  return {
    available: true,
    rising,
    falling,
    stable,
    raw_data: momentum,
  };
}

/**
 * Get naming events - moments where identities were established
 */
export async function handleIdentityGetNamingEvents({ limit }: { limit?: number }) {
  const records = loadIdentityAnalysis();
  const namingRecords = records.filter((r) => r.type === "identity.naming_event");

  if (namingRecords.length === 0) {
    return {
      available: false,
      message: "No naming events found. Run analyze_identity.py first.",
    };
  }

  const events = namingRecords.map((r) => r.data as NamingEvent);
  const limitedEvents = limit != null ? events.slice(0, limit) : events;

  // Group by name
  const byName: Record<string, NamingEvent[]> = {};
  for (const event of events) {
    if (!byName[event.name]) {
      byName[event.name] = [];
    }
    byName[event.name].push(event);
  }

  // Sort names by frequency
  const topNames = Object.entries(byName)
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 10)
    .map(([name, evts]) => ({ name, count: evts.length }));

  return {
    available: true,
    total_events: events.length,
    top_names: topNames,
    events: limitedEvents,
  };
}

/**
 * Get co-occurrence clusters - concepts that appear together
 */
export async function handleIdentityGetClusters({ min_count }: { min_count?: number }) {
  const clusters = getRecordByType("identity.clusters") as Record<string, number> | null;

  if (!clusters) {
    return {
      available: false,
      message: "Cluster data not found. Run analyze_identity.py first.",
    };
  }

  const minThreshold = min_count ?? 0;
  const filtered = Object.entries(clusters)
    .filter(([, count]) => count >= minThreshold)
    .sort((a, b) => b[1] - a[1]);

  return {
    available: true,
    total_clusters: Object.keys(clusters).length,
    clusters: Object.fromEntries(filtered),
  };
}

/**
 * Get relational patterns - we/I ratios, role language
 */
export async function handleIdentityGetRelational() {
  const relational = getRecordByType("identity.relational") as Record<string, unknown> | null;

  if (!relational) {
    return {
      available: false,
      message: "Relational data not found. Run analyze_identity.py first.",
    };
  }

  return {
    available: true,
    ...relational,
    interpretation: interpretRelational(relational),
  };
}

function interpretRelational(data: Record<string, unknown>): string[] {
  const insights: string[] = [];
  
  const weIUser = data.we_vs_i_ratio_user as number;
  const weIAssistant = data.we_vs_i_ratio_assistant as number;

  if (weIUser > 0.5) {
    insights.push("User frequently uses collaborative language (we/us)");
  } else if (weIUser < 0.2) {
    insights.push("User primarily uses individual language (I/me)");
  }

  if (weIAssistant > 0.5) {
    insights.push("Assistant uses collaborative framing, suggesting partnership");
  }

  if (weIAssistant > weIUser * 1.5) {
    insights.push("Assistant uses more collaborative language than user, possibly building rapport");
  }

  return insights;
}

/**
 * Clear the cached analysis (useful after regenerating)
 */
export function clearIdentityAnalysisCache() {
  cachedAnalysis = null;
}

