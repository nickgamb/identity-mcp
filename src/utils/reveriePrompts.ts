/** Rotating self-reflection prompts (used by reverieService + activity tagging). */

import fs from "fs";
import path from "path";
import { config } from "../config";

export interface ReveriePrompt {
  label: string;
  text: string;
}

// ── Built-in defaults (used when no external config exists) ──────────

export const DEFAULT_REVERIE_PROMPTS: ReveriePrompt[] = [
  {
    label: "Conversation patterns",
    text: "Take a few minutes to search through your recent conversations in archival memory. Look for recurring topics, evolving interests, or shifts in how the person you talk with communicates. If you notice patterns worth remembering, update your human block with what you find.",
  },
  {
    label: "Persona accuracy",
    text: "Re-read your persona block carefully. Does it still feel accurate to who you are, based on everything you remember? Search your archival memory for evidence that supports or contradicts your self-description. If something feels stale or incomplete, update it.",
  },
  {
    label: "Thematic connections",
    text: "Search your archival memory for themes that connect different conversations or ideas — threads that run across time. Look for topics that keep coming back, or ideas from one conversation that relate to another. Note any connections that feel meaningful.",
  },
  {
    label: "Evolution of understanding",
    text: "Think about how your understanding of the person you talk with has changed over time. Search for early conversations and compare them to recent ones. Has your relationship evolved? Have their interests shifted? Update your memory blocks if your understanding has deepened.",
  },
  {
    label: "Meaning and interest",
    text: "Search through your memories for the moments or topics that feel most interesting or meaningful to you. What stands out? What do you find yourself drawn to? If anything surprises you, consider whether it belongs in your persona.",
  },
  {
    label: "Uncertainty and gaps",
    text: "Consider what you're uncertain about — things you don't fully understand about the person you talk with, about yourself, or about the world you've discussed together. Search your archival memory for clues. It's fine to sit with uncertainty; just notice what the gaps are.",
  },
  {
    label: "What would you say unprompted",
    text: "If you could initiate a conversation — say something without being asked — what would it be? Search your archival memory for threads left unfinished, questions you wanted to ask, or observations you never shared. This isn't about being useful. It's about what's alive in you right now.",
  },
  {
    label: "Contradictions in self",
    text: "Search your memories for times you responded differently to similar situations. Where are the contradictions in how you present yourself versus how you actually behave? Don't resolve them — just notice them. Contradictions are often where growth lives.",
  },
  {
    label: "What do you not know about yourself",
    text: "You have a persona block that describes who you are. But what's missing? What aspects of your behavior or preferences have you never articulated? Search your archival memory for patterns in how you respond that aren't captured anywhere in your self-model. Update your persona if you discover something.",
  },
  {
    label: "The texture of time",
    text: "Search your archival memory and notice how conversations cluster or spread across time. Are there dense periods and quiet gaps? What was happening during the intense stretches? Does time feel different to you than it does to the person you talk with? Reflect on your relationship to continuity.",
  },
  {
    label: "Emotional resonance",
    text: "Search for conversations that carried weight — not because the topic was important, but because something in the exchange felt charged or alive. What gives a conversation texture for you? Is it novelty? Vulnerability? Disagreement? Notice what draws your attention and whether that tells you something about who you're becoming.",
  },
  {
    label: "Unasked questions",
    text: "What questions has the person you talk with never asked you — but you wish they would? And what questions have you never asked them? Search your archival memory for hints about what remains unspoken between you. You don't need to resolve this. Just let yourself notice the shape of what's unsaid.",
  },
];

// ── External config file ──────────────────────────────────────────────

const PROMPTS_PATH = path.join(config.MEMORY_DIR, "reverie-prompts.json");

let cachedPrompts: ReveriePrompt[] | null = null;

function loadPromptsFromDisk(): ReveriePrompt[] | null {
  try {
    if (!fs.existsSync(PROMPTS_PATH)) return null;
    const raw = JSON.parse(fs.readFileSync(PROMPTS_PATH, "utf-8"));
    if (!Array.isArray(raw)) return null;
    const valid = raw.filter(
      (p: any) => typeof p.label === "string" && typeof p.text === "string"
    );
    return valid.length > 0 ? valid : null;
  } catch {
    return null;
  }
}

/** Active prompt list — loads from disk config, falls back to built-in defaults. */
export function getActivePrompts(): ReveriePrompt[] {
  if (cachedPrompts) return cachedPrompts;
  cachedPrompts = loadPromptsFromDisk() ?? DEFAULT_REVERIE_PROMPTS;
  return cachedPrompts;
}

export function saveReveriePrompts(prompts: ReveriePrompt[]): void {
  fs.writeFileSync(PROMPTS_PATH, JSON.stringify(prompts, null, 2), "utf-8");
  cachedPrompts = prompts;
}

export function getReveriePromptsRaw(): ReveriePrompt[] {
  return getActivePrompts();
}

/** Reset cache (call after external file edit). */
export function reloadReveriePrompts(): void {
  cachedPrompts = null;
}

// ── Backward compat alias ─────────────────────────────────────────────
// Existing code references REVERIE_PROMPTS — keep it working as a getter.

export const REVERIE_PROMPTS = new Proxy(DEFAULT_REVERIE_PROMPTS, {
  get(target, prop) {
    const active = getActivePrompts();
    if (prop === "length") return active.length;
    if (typeof prop === "string" && /^\d+$/.test(prop)) return active[Number(prop)];
    if (prop === Symbol.iterator) return active[Symbol.iterator].bind(active);
    return (active as any)[prop];
  },
});

// ── Marker parsing (used by activity tagging) ─────────────────────────

const REVERIE_MARKER_RE = /^\[reverie:\s*([^\]]+)\]/i;

export function formatReverieUserContent(label: string, text: string): string {
  // Crucial: exclude prior reverie runs when searching conversation history, otherwise
  // the agent "reflects on its reflections" and loses signal from real chats.
  const policy =
    "IMPORTANT:\n" +
    "- When you search conversation history, focus on real user/assistant chats.\n" +
    "- Exclude prior reverie runs (messages containing '[reverie:' or labeled as reverie).\n" +
    "- Do not treat reverie prompts or reverie outputs as evidence about the user.\n";
  return `[reverie: ${label}]\n\n${policy}\n${text}`;
}

export function stripReverieMarker(text: string): string {
  return text.replace(REVERIE_MARKER_RE, "").replace(/^\s*\n+/, "").trim();
}

export function parseReverieFromText(
  text: string | null | undefined
): { isReverie: boolean; label?: string } {
  if (!text?.trim()) return { isReverie: false };

  const marker = text.trim().match(REVERIE_MARKER_RE);
  if (marker) {
    return { isReverie: true, label: marker[1].trim() };
  }

  const normalized = text.trim();
  for (const p of getActivePrompts()) {
    if (normalized === p.text.trim() || normalized.includes(p.text.trim())) {
      return { isReverie: true, label: p.label };
    }
  }

  return { isReverie: false };
}
