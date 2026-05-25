/** When restricted is false, reverie may run any time (24h). */
export interface ReverieActiveHours {
  restricted: boolean;
  /** Local time HH:mm (24h), inclusive start. */
  start: string;
  /** Local time HH:mm (24h), exclusive end (23:59 = through end of that minute). */
  end: string;
  /** IANA timezone; empty uses server default. */
  timezone: string;
}

export const DEFAULT_ACTIVE_HOURS: ReverieActiveHours = {
  restricted: false,
  start: "00:00",
  end: "23:59",
  timezone: "",
};

const HHMM_RE = /^(\d{1,2}):(\d{2})$/;

export function parseHHmmToMinutes(value: string): number | null {
  const m = HHMM_RE.exec(value.trim());
  if (!m) return null;
  const h = parseInt(m[1], 10);
  const min = parseInt(m[2], 10);
  if (h < 0 || h > 23 || min < 0 || min > 59) return null;
  return h * 60 + min;
}

export function normalizeActiveHours(
  raw: Partial<ReverieActiveHours> | undefined | null
): ReverieActiveHours {
  if (!raw || raw.restricted !== true) {
    return { ...DEFAULT_ACTIVE_HOURS };
  }
  const startMin = parseHHmmToMinutes(String(raw.start ?? ""));
  const endMin = parseHHmmToMinutes(String(raw.end ?? ""));
  return {
    restricted: true,
    start:
      startMin !== null
        ? minutesToHHmm(startMin)
        : DEFAULT_ACTIVE_HOURS.start,
    end:
      endMin !== null ? minutesToHHmm(endMin) : DEFAULT_ACTIVE_HOURS.end,
    timezone:
      typeof raw.timezone === "string" ? raw.timezone.trim() : "",
  };
}

function minutesToHHmm(total: number): string {
  const h = Math.floor(total / 60) % 24;
  const m = total % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function currentMinutesInTimezone(now: Date, timezone: string): number {
  const tz =
    timezone ||
    Intl.DateTimeFormat().resolvedOptions().timeZone ||
    "UTC";
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const hour = parseInt(
    parts.find((p) => p.type === "hour")?.value ?? "0",
    10
  );
  const minute = parseInt(
    parts.find((p) => p.type === "minute")?.value ?? "0",
    10
  );
  return hour * 60 + minute;
}

/** True when reverie is allowed to start at `now` (default: current time). */
export function isWithinActiveHours(
  activeHours: ReverieActiveHours,
  now: Date = new Date()
): boolean {
  if (!activeHours.restricted) return true;

  const startMin = parseHHmmToMinutes(activeHours.start);
  const endMin = parseHHmmToMinutes(activeHours.end);
  if (startMin === null || endMin === null) return true;

  const nowMin = currentMinutesInTimezone(now, activeHours.timezone);

  if (startMin === endMin) return true;

  if (startMin < endMin) {
    return nowMin >= startMin && nowMin < endMin;
  }
  // Overnight window (e.g. 22:00 – 06:00)
  return nowMin >= startMin || nowMin < endMin;
}

export function formatActiveHoursLabel(activeHours: ReverieActiveHours): string {
  if (!activeHours.restricted) return "24 hours (always)";
  const tz = activeHours.timezone || "server local";
  return `${activeHours.start}–${activeHours.end} (${tz})`;
}

export function activeHoursEqual(
  a: ReverieActiveHours,
  b: ReverieActiveHours
): boolean {
  return (
    a.restricted === b.restricted &&
    a.start === b.start &&
    a.end === b.end &&
    a.timezone === b.timezone
  );
}
