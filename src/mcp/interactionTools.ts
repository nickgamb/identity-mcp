import fs from "fs";
import path from "path";
import { config } from "../config";
import { logger } from "../utils/logger";
import { getUserDataPath } from "../utils/userContext";

interface InteractionEvent {
  event_type: string;
  conversation_id: string;
  file_path: string;
  timestamp: number | null;
  short_description: string;
}

interface ConversationIndex {
  conversation_id: string;
  file_path: string;
  earliest_timestamp: number | null;
  latest_timestamp: number | null;
  message_count: number;
  user_message_count: number;
  topic_tags: string[];
  tone_tags: string[];
  truncated_or_corrupted: boolean;
  messages_preview: Array<{
    role: string;
    timestamp: string;
    content_preview: string;
  }>;
}

// Note: Cache is per-process, not per-user. For multi-user, we'd need per-user cache.
let cachedEvents: InteractionEvent[] | null = null;
let cachedIndex: ConversationIndex[] | null = null;
let cachedEventsUserId: string | null = null;
let cachedIndexUserId: string | null = null;

function getMemoryDir(userId: string | null = null): string {
  return getUserDataPath(config.MEMORY_DIR, userId);
}

function loadKeyEvents(userId: string | null = null): InteractionEvent[] {
  // Use cache only if same user or no user (single-user mode)
  if (cachedEvents && (!userId || cachedEventsUserId === userId)) {
    return cachedEvents;
  }

  const filePath = path.join(getMemoryDir(userId), "interaction_key_events.json");

  if (!fs.existsSync(filePath)) {
    logger.warn("Interaction key events file not found", { path: filePath });
    return [];
  }

  try {
    const content = fs.readFileSync(filePath, "utf8");
    const events = JSON.parse(content);
    // Cache only in single-user mode or if same user
    if (!userId || cachedEventsUserId === userId) {
      cachedEvents = events;
      cachedEventsUserId = userId;
    }
    logger.info("Loaded interaction key events", { count: events?.length, userId });
    return events || [];
  } catch (error) {
    logger.error("Failed to load interaction key events", { error: String(error) });
    return [];
  }
}

function loadIndex(userId: string | null = null): ConversationIndex[] {
  // Use cache only if same user or no user (single-user mode)
  if (cachedIndex && (!userId || cachedIndexUserId === userId)) {
    return cachedIndex;
  }

  const filePath = path.join(getMemoryDir(userId), "interaction_map_index.json");

  if (!fs.existsSync(filePath)) {
    logger.warn("Interaction map index file not found", { path: filePath });
    return [];
  }

  try {
    const content = fs.readFileSync(filePath, "utf8");
    const index = JSON.parse(content);
    // Cache only in single-user mode or if same user
    if (!userId || cachedIndexUserId === userId) {
      cachedIndex = index;
      cachedIndexUserId = userId;
    }
    logger.info("Loaded interaction map index", { count: index?.length, userId });
    return index || [];
  } catch (error) {
    logger.error("Failed to load interaction map index", { error: String(error) });
    return [];
  }
}

/**
 * Get summary of interaction data
 */
export async function handleInteractionGetSummary(userId: string | null = null) {
  const events = loadKeyEvents(userId);
  const index = loadIndex(userId);

  if (events.length === 0 && index.length === 0) {
    return {
      available: false,
      message: "Interaction data not found. Run build_interaction_map.py first.",
    };
  }

  // Count event types
  const eventTypes: Record<string, number> = {};
  for (const event of events) {
    eventTypes[event.event_type] = (eventTypes[event.event_type] || 0) + 1;
  }

  // Count conversations by topic/tone
  const topicCounts: Record<string, number> = {};
  const toneCounts: Record<string, number> = {};
  for (const conv of index) {
    for (const topic of conv.topic_tags) {
      topicCounts[topic] = (topicCounts[topic] || 0) + 1;
    }
    for (const tone of conv.tone_tags) {
      toneCounts[tone] = (toneCounts[tone] || 0) + 1;
    }
  }

  return {
    available: true,
    total_conversations: index.length,
    total_human_messages: index.reduce((sum, c) => sum + c.user_message_count, 0),
    total_key_events: events.length,
    event_types: eventTypes,
    topic_distribution: topicCounts,
    tone_distribution: toneCounts,
  };
}

/**
 * Get key events by type
 */
export async function handleInteractionGetEvents({
  event_type,
  limit,
}: {
  event_type?: string;
  limit?: number;
}, userId: string | null = null) {
  const events = loadKeyEvents(userId);

  if (events.length === 0) {
    return {
      available: false,
      message: "No key events found. Run build_interaction_map.py first.",
    };
  }

  let filtered = events;

  if (event_type) {
    filtered = events.filter((e) => e.event_type === event_type);
  }

  // Sort by timestamp (most recent first)
  filtered.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

  const limited = limit != null ? filtered.slice(0, limit) : filtered;

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
 * Search conversations by topic, tone, or keyword
 */
export async function handleInteractionSearch({
  query,
  limit,
}: {
  query: string;
  limit?: number;
}, userId: string | null = null) {
  const index = loadIndex(userId);

  if (index.length === 0) {
    return {
      available: false,
      message: "Interaction index not found. Run build_interaction_map.py first.",
    };
  }

  const queryLower = query.toLowerCase();

  // Search in topic tags, tone tags, and message previews
  const matches = index.filter((conv) => {
    // Check topic tags
    if (conv.topic_tags.some((tag) => tag.toLowerCase().includes(queryLower))) {
      return true;
    }
    // Check tone tags
    if (conv.tone_tags.some((tag) => tag.toLowerCase().includes(queryLower))) {
      return true;
    }
    // Check message previews
    if (conv.messages_preview.some((msg) => msg.content_preview.toLowerCase().includes(queryLower))) {
      return true;
    }
    return false;
  });

  const limited = limit != null ? matches.slice(0, limit) : matches.slice(0, 20);

  return {
    available: true,
    query,
    total_matches: matches.length,
    results: limited.map((conv) => ({
      conversation_id: conv.conversation_id,
      file_path: conv.file_path,
      message_count: conv.message_count,
      user_message_count: conv.user_message_count,
      topic_tags: conv.topic_tags,
      tone_tags: conv.tone_tags,
    })),
  };
}

/**
 * Get conversations by topic
 */
export async function handleInteractionGetByTopic({
  topic,
  limit,
}: {
  topic: string;
  limit?: number;
}, userId: string | null = null) {
  const index = loadIndex(userId);

  if (index.length === 0) {
    return {
      available: false,
      message: "Interaction index not found. Run build_interaction_map.py first.",
    };
  }

  const topicLower = topic.toLowerCase();
  const matches = index.filter((conv) =>
    conv.topic_tags.some((tag) => tag.toLowerCase() === topicLower)
  );

  const limited = matches.slice(0, limit ?? 20);

  return {
    available: true,
    topic,
    total_matches: matches.length,
    conversations: limited.map((conv) => ({
      conversation_id: conv.conversation_id,
      file_path: conv.file_path,
      message_count: conv.message_count,
      user_message_count: conv.user_message_count,
      topic_tags: conv.topic_tags,
      tone_tags: conv.tone_tags,
    })),
  };
}

/**
 * Get timeline of key events
 */
export async function handleInteractionGetTimeline({
  start_date,
  end_date,
}: {
  start_date?: string;
  end_date?: string;
}, userId: string | null = null) {
  const events = loadKeyEvents(userId);

  if (events.length === 0) {
    return {
      available: false,
      message: "No key events found. Run build_interaction_map.py first.",
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
  const byMonth: Record<string, InteractionEvent[]> = {};
  for (const event of filtered) {
    if (event.timestamp) {
      const date = new Date(event.timestamp * 1000);
      const month = date.toISOString().substring(0, 7); // YYYY-MM
      if (!byMonth[month]) {
        byMonth[month] = [];
      }
      byMonth[month].push(event);
    }
  }

  return {
    available: true,
    total_events: filtered.length,
    by_month: Object.fromEntries(
      Object.entries(byMonth).map(([month, evts]) => [
        month,
        { count: evts.length, event_types: [...new Set(evts.map((e) => e.event_type))] },
      ])
    ),
  };
}

/**
 * Clear cached interaction data
 */
export function clearInteractionCache() {
  cachedEvents = null;
  cachedIndex = null;
}

