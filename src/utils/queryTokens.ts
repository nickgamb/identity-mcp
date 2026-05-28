/**
 * Tokenize a search query for local (non-semantic) text matching.
 *
 * Reverie / agent queries are usually concatenated keyword strings like
 * "daemon shadow abyss origin codex" — meant for a vector store. Matching
 * those as one literal substring returns nothing. We split into individual
 * lowercased tokens, drop stopwords/short noise, and score by hit count.
 */

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
  "have", "he", "her", "him", "his", "how", "i", "in", "is", "it", "its",
  "me", "my", "of", "on", "or", "she", "so", "than", "that", "the", "their",
  "them", "they", "this", "to", "us", "was", "we", "were", "what", "when",
  "where", "which", "who", "why", "will", "with", "you", "your",
]);

const TOKEN_SPLIT = /[\s,.;:!?'"()\[\]{}<>/\\|`~@#$%^&*+=]+/;

export interface QueryMatcher {
  /** Lowercased tokens used for matching (deduped, stopwords removed). */
  tokens: string[];
  /** Original lowercased+trimmed query for legacy exact-substring checks. */
  raw: string;
  /** Empty when no usable tokens (caller should short-circuit). */
  isEmpty: boolean;
  /** Count of distinct tokens found in text — use for relevance scoring. */
  matchCount(text: string): number;
  /** True if text contains any token. */
  matches(text: string): boolean;
}

export function tokenize(query: string): QueryMatcher {
  const raw = query.toLowerCase().trim();
  const tokens = Array.from(
    new Set(
      raw
        .split(TOKEN_SPLIT)
        .filter((t) => t.length >= 3 && !STOPWORDS.has(t))
    )
  );

  return {
    tokens,
    raw,
    isEmpty: tokens.length === 0,
    matchCount(text: string): number {
      if (tokens.length === 0 || !text) return 0;
      const lower = text.toLowerCase();
      let hits = 0;
      for (const tok of tokens) {
        if (lower.includes(tok)) hits++;
      }
      return hits;
    },
    matches(text: string): boolean {
      if (tokens.length === 0 || !text) return false;
      const lower = text.toLowerCase();
      for (const tok of tokens) {
        if (lower.includes(tok)) return true;
      }
      return false;
    },
  };
}
