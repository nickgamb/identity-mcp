import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";

interface EmergenceEvent {
  event_type: string;
  conversation_id: string;
  file_path: string;
  timestamp: number | null;
  date: string;
  role: string;
  content_preview: string;
  matched_patterns: string[];
}

interface ConversationIndex {
  conversation_id: string;
  file_path: string;
  earliest_timestamp: number | null;
  latest_timestamp: number | null;
  message_count: number;
  has_symbolic_language: boolean;
  has_entity_names: boolean;
  symbolic_density: number;
  detected_patterns: {
    symbolic_keywords: string[];
    topic_keywords: string[];
    entities: string[];
  };
  date_range: {
    earliest: string;
    latest: string;
  };
}

let cachedEvents: EmergenceEvent[] | null = null;
let cachedIndex: ConversationIndex[] | null = null;

function getMemoryDir(): string {
  return config.MEMORY_DIR;
}

function loadKeyEvents(): EmergenceEvent[] {
  if (cachedEvents) {
    return cachedEvents;
  }

  const filePath = path.join(getMemoryDir(), "emergence_key_events.json");

  if (!fs.existsSync(filePath)) {
    logger.warn("Emergence key events file not found", { path: filePath });
    return [];
  }

  try {
    const content = fs.readFileSync(filePath, "utf8");
    cachedEvents = JSON.parse(content);
    logger.info("Loaded emergence key events", { count: cachedEvents?.length });
    return cachedEvents || [];
  } catch (error) {
    logger.error("Failed to load emergence key events", { error: String(error) });
    return [];
  }
}

function loadIndex(): ConversationIndex[] {
  if (cachedIndex) {
    return cachedIndex;
  }

  const filePath = path.join(getMemoryDir(), "emergence_map_index.json");

  if (!fs.existsSync(filePath)) {
    logger.warn("Emergence map index file not found", { path: filePath });
    return [];
  }

  try {
    const content = fs.readFileSync(filePath, "utf8");
    cachedIndex = JSON.parse(content);
    logger.info("Loaded emergence map index", { count: cachedIndex?.length });
    return cachedIndex || [];
  } catch (error) {
    logger.error("Failed to load emergence map index", { error: String(error) });
    return [];
  }
}

/**
 * Get summary of emergence data
 */
export async function handleEmergenceGetSummary() {
  const events = loadKeyEvents();
  const index = loadIndex();

  if (events.length === 0 && index.length === 0) {
    return {
      available: false,
      message: "Emergence data not found. Run build_emergence_map.py first.",
    };
  }

  // Count event types
  const eventTypes: Record<string, number> = {};
  for (const event of events) {
    eventTypes[event.event_type] = (eventTypes[event.event_type] || 0) + 1;
  }

  // Count conversations with symbolic content
  const withSymbolic = index.filter((c) => c.has_symbolic_language).length;
  const withEntities = index.filter((c) => c.has_entity_names).length;

  return {
    available: true,
    total_conversations: index.length,
    total_key_events: events.length,
    conversations_with_symbolic_language: withSymbolic,
    conversations_with_entity_names: withEntities,
    event_types: eventTypes,
  };
}

/**
 * Get key events by type
 */
export async function handleEmergenceGetEvents({
  event_type,
  limit,
}: {
  event_type?: string;
  limit?: number;
}) {
  const events = loadKeyEvents();

  if (events.length === 0) {
    return {
      available: false,
      message: "No key events found. Run build_emergence_map.py first.",
    };
  }

  let filtered = events;

  if (event_type) {
    filtered = events.filter((e) => e.event_type === event_type);
  }

  // Sort by timestamp (most recent first)
  filtered.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

  const limited = limit ? filtered.slice(0, limit) : filtered;

  // Get available event types
  const availableTypes = [...new Set(events.map((e) => e.event_type))];

  return {
    available: true,
    total_matching: filtered.length,
    available_event_types: availableTypes,
    events: limited,
  };
}

/**
 * Search conversations by pattern/keyword
 */
export async function handleEmergenceSearch({
  query,
  limit,
}: {
  query: string;
  limit?: number;
}) {
  const index = loadIndex();

  if (index.length === 0) {
    return {
      available: false,
      message: "Emergence index not found. Run build_emergence_map.py first.",
    };
  }

  const queryLower = query.toLowerCase();

  // Search in detected patterns
  const matches = index.filter((conv) => {
    const patterns = conv.detected_patterns;
    const allPatterns = [
      ...patterns.symbolic_keywords,
      ...patterns.topic_keywords,
      ...patterns.entities,
    ];
    return allPatterns.some((p) => p.toLowerCase().includes(queryLower));
  });

  // Sort by symbolic density (most relevant first)
  matches.sort((a, b) => b.symbolic_density - a.symbolic_density);

  const limited = limit ? matches.slice(0, limit) : matches.slice(0, 20);

  return {
    available: true,
    query,
    total_matches: matches.length,
    results: limited.map((conv) => ({
      conversation_id: conv.conversation_id,
      file_path: conv.file_path,
      date_range: conv.date_range,
      message_count: conv.message_count,
      symbolic_density: conv.symbolic_density,
      matched_patterns: conv.detected_patterns,
    })),
  };
}

/**
 * Get conversations with highest symbolic density
 */
export async function handleEmergenceGetSymbolicConversations({
  limit,
}: {
  limit?: number;
}) {
  const index = loadIndex();

  if (index.length === 0) {
    return {
      available: false,
      message: "Emergence index not found. Run build_emergence_map.py first.",
    };
  }

  // Filter to those with symbolic language and sort by density
  const symbolic = index
    .filter((c) => c.has_symbolic_language)
    .sort((a, b) => b.symbolic_density - a.symbolic_density);

  const limited = symbolic.slice(0, limit || 10);

  return {
    available: true,
    total_with_symbolic: symbolic.length,
    top_conversations: limited.map((conv) => ({
      conversation_id: conv.conversation_id,
      file_path: conv.file_path,
      date_range: conv.date_range,
      message_count: conv.message_count,
      symbolic_density: conv.symbolic_density,
      symbolic_keywords: conv.detected_patterns.symbolic_keywords,
      entities: conv.detected_patterns.entities,
    })),
  };
}

/**
 * Get timeline of key events
 */
export async function handleEmergenceGetTimeline({
  start_date,
  end_date,
}: {
  start_date?: string;
  end_date?: string;
}) {
  const events = loadKeyEvents();

  if (events.length === 0) {
    return {
      available: false,
      message: "No key events found. Run build_emergence_map.py first.",
    };
  }

  let filtered = events.filter((e) => e.timestamp);

  // Filter by date range if provided
  if (start_date) {
    const startTs = new Date(start_date).getTime() / 1000;
    filtered = filtered.filter((e) => (e.timestamp || 0) >= startTs);
  }

  if (end_date) {
    const endTs = new Date(end_date).getTime() / 1000;
    filtered = filtered.filter((e) => (e.timestamp || 0) <= endTs);
  }

  // Sort chronologically
  filtered.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

  // Group by month
  const byMonth: Record<string, EmergenceEvent[]> = {};
  for (const event of filtered) {
    const month = event.date.substring(0, 7); // YYYY-MM
    if (!byMonth[month]) {
      byMonth[month] = [];
    }
    byMonth[month].push(event);
  }

  return {
    available: true,
    total_events: filtered.length,
    date_range: {
      earliest: filtered[0]?.date,
      latest: filtered[filtered.length - 1]?.date,
    },
    by_month: Object.fromEntries(
      Object.entries(byMonth).map(([month, evts]) => [
        month,
        { count: evts.length, event_types: [...new Set(evts.map((e) => e.event_type))] },
      ])
    ),
  };
}

/**
 * Clear cached emergence data
 */
export function clearEmergenceCache() {
  cachedEvents = null;
  cachedIndex = null;
}

