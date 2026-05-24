/**
 * Classify and date archival passages — shared by lettaProxy and dashboard.
 * Headers are produced by letta/ingest.py.
 */

export type ArchivalPassageType =
  | "conversation"
  | "file"
  | "memory"
  | "analysis"
  | "other";

const MEMORY_KINDS = new Set([
  "chatgpt_memory",
  "claude_memory",
  "memory",
  "memory_json",
  "memory_report",
  "user.context",
]);

function parseHeader(text: string): {
  cat: string;
  kind: string;
  source: string;
} | null {
  if (!text.startsWith("[")) return null;
  const bracket = text.split("]")[0].replace("[", "");
  const parts = bracket.split("|").map((s) => s.trim());
  if (parts.length < 2) return null;
  return {
    cat: parts[0],
    kind: parts[1].split(/\s/)[0],
    source: parts[2] ?? "",
  };
}

/** Classify passage type from ingest header prefix. */
export function classifyArchivalPassage(text: string): ArchivalPassageType {
  const h = parseHeader(text);
  if (!h) return "other";

  const { cat, kind, source } = h;
  const isDate = /^\d{4}-\d{2}-\d{2}/.test(cat);

  // Model outputs: [file | models/identity/...] or [file | models/eeg_identity/...]
  if (
    cat === "file" &&
    (kind.startsWith("models/") ||
      source.includes("models/identity") ||
      source.includes("models/eeg_identity"))
  ) {
    return "analysis";
  }

  if (cat === "file" || cat === "tabular") return "file";

  if (kind === "conversation" && isDate) return "conversation";

  if (
    MEMORY_KINDS.has(kind) ||
    MEMORY_KINDS.has(cat) ||
    (isDate && kind !== "conversation")
  ) {
    return "memory";
  }

  if (
    kind.startsWith("identity") ||
    kind.startsWith("pattern") ||
    cat.startsWith("identity") ||
    cat.startsWith("pattern")
  ) {
    return "analysis";
  }

  if (cat === "undated" && MEMORY_KINDS.has(kind)) return "memory";

  return "other";
}

/** ISO date YYYY-MM-DD from header or created_at (not ingest timestamp when header has date). */
export function passageDateKey(
  text: string,
  createdAt?: string | null
): string | null {
  const headerMatch = text.match(/^\[(\d{4}-\d{2}-\d{2})/);
  if (headerMatch) return headerMatch[1];

  if (!createdAt) return null;
  const d = new Date(createdAt);
  if (Number.isNaN(d.getTime())) {
    if (/^\d{4}-\d{2}-\d{2}/.test(createdAt)) return createdAt.slice(0, 10);
    return null;
  }
  return d.toISOString().slice(0, 10);
}

export function passageMatchesDateRange(
  text: string,
  createdAt: string | undefined | null,
  dateFrom?: string,
  dateTo?: string
): boolean {
  if (!dateFrom && !dateTo) return true;
  const key = passageDateKey(text, createdAt);
  if (!key) return false;
  if (dateFrom && key < dateFrom) return false;
  if (dateTo && key > dateTo) return false;
  return true;
}

export function matchesArchivalTypeFilter(
  text: string,
  filter: ArchivalPassageType
): boolean {
  return classifyArchivalPassage(text) === filter;
}

/** Non-conversation types are ingested last — scan from newest end of archival memory. */
export function preferNewestScanForType(
  typeFilter?: ArchivalPassageType
): boolean {
  return !!typeFilter && typeFilter !== "conversation";
}
