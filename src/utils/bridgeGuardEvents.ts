import fs from "fs";
import path from "path";
import { config } from "../config.js";
import type { LettaMessage } from "../mcp/lettaProxy.js";

const GUARD_LOG_FILE = "bridge-guard-events.jsonl";
const DEFAULT_TAIL = 80;

export interface BridgeGuardEventRow {
  ts: string;
  kind: string;
  reason: string;
  source?: string;
  agent_id?: string;
  run_id?: string;
  total_chars?: number;
  sample?: string;
  pattern?: string;
  marker_count?: number;
}

function guardLogPath(): string {
  return path.join(config.MEMORY_DIR, GUARD_LOG_FILE);
}

function formatGuardContent(row: BridgeGuardEventRow): string {
  const lines = [
    `Reasoning loop detected (${row.reason})`,
    row.source ? `Source: ${row.source}` : null,
    row.run_id ? `Run: ${row.run_id}` : null,
    row.total_chars != null ? `Planning chars before trim: ${row.total_chars}` : null,
    row.pattern ? `Repeating pattern:\n${row.pattern}` : null,
    row.marker_count != null ? `Meta-token lines in window: ${row.marker_count}` : null,
    row.sample ? `Raw tail that triggered the guard:\n${row.sample}` : null,
  ].filter(Boolean);
  return lines.join("\n\n");
}

/** Read recent bridge guard events as Activity-tab messages (newest first). */
export function loadBridgeGuardActivity(limit = DEFAULT_TAIL): LettaMessage[] {
  const filePath = guardLogPath();
  if (!fs.existsSync(filePath)) {
    return [];
  }

  let raw: string;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch {
    return [];
  }

  const rows: BridgeGuardEventRow[] = [];
  for (const line of raw.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    try {
      rows.push(JSON.parse(t) as BridgeGuardEventRow);
    } catch {
      /* skip corrupt line */
    }
  }

  return rows
    .slice(-limit)
    .reverse()
    .map((row, i) => ({
      id: `bridge-guard-${row.ts}-${i}`,
      role: "bridge_guard",
      message_type: "bridge_guard_event",
      event_type: row.reason.toUpperCase(),
      content: formatGuardContent(row),
      created_at: row.ts,
      run_id: row.run_id,
    }));
}
