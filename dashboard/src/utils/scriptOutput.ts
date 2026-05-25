/** Keep terminal UI responsive during long runs (tqdm, ingest logs). */
export const MAX_TERMINAL_LINES = 4000;

export function appendOutputLines(prev: string[], newLines: string[]): string[] {
  if (newLines.length === 0) return prev;
  let next = prev.concat(newLines);
  if (next.length > MAX_TERMINAL_LINES) {
    const drop = next.length - MAX_TERMINAL_LINES;
    next = next.slice(drop);
    if (!next[0]?.startsWith('[...')) {
      next = [`[... ${drop} earlier lines truncated in UI ...]`, ...next];
    }
  }
  return next;
}
