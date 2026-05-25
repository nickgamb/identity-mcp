import { useMemo } from 'react'
import { ExternalLink } from 'lucide-react'

/** Browser-reachable LibreChat URL (direct :3080 — not proxied). */
function libreChatUrl(): string {
  const fromEnv = import.meta.env.VITE_LIBRECHAT_URL?.trim()
  if (fromEnv) return fromEnv.replace(/\/$/, '')
  return `${window.location.protocol}//${window.location.hostname}:3080`
}

export function LibreChatEmbed() {
  const src = useMemo(() => libreChatUrl(), [])

  return (
    <div className="flex flex-col flex-1 min-h-0 bg-surface">
      <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-surface-200 bg-surface-50/80 shrink-0">
        <p className="text-xs text-text-muted">
          Embedded Chat: {' '}
          <a
            href={src}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:text-accent-bright font-mono"
          >
            {src}
          </a>
        </p>
        <a
          href={src}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-ghost text-xs shrink-0"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          Open in new tab
        </a>
      </div>
      <iframe
        title="LibreChat"
        src={src}
        className="flex-1 w-full min-h-0 border-0 bg-surface"
        allow="clipboard-read; clipboard-write; microphone"
      />
    </div>
  )
}
