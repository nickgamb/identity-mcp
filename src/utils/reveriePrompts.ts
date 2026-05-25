/** Rotating self-reflection prompts (used by reverieService + activity tagging). */

export interface ReveriePrompt {
  label: string;
  text: string;
}

export const REVERIE_PROMPTS: ReveriePrompt[] = [
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
];

const REVERIE_MARKER_RE = /^\[reverie:\s*([^\]]+)\]/i;

/** Prefix sent with reverie user messages so Activity can tag them. */
export function formatReverieUserContent(label: string, text: string): string {
  return `[reverie: ${label}]\n\n${text}`;
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
  for (const p of REVERIE_PROMPTS) {
    if (normalized === p.text.trim() || normalized.includes(p.text.trim())) {
      return { isReverie: true, label: p.label };
    }
  }

  return { isReverie: false };
}
