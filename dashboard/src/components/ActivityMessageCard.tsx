import { ChevronDown, ChevronUp, Copy, Check, Clock, Moon, AlertCircle } from 'lucide-react'
import {
  type ActivityMessage,
  activityBodyText,
  activityRoleConfig,
  isSleeptimeActivity,
  PREVIEW_CHARS,
} from '../utils/lettaActivity'

interface ActivityMessageCardProps {
  message: ActivityMessage
  index: number
  expanded: boolean
  copiedId: string | null
  onToggleExpand: () => void
  onCopy: (text: string, id: string) => void
}

export function ActivityMessageCard({
  message: m,
  index,
  expanded,
  copiedId,
  onToggleExpand,
  onCopy,
}: ActivityMessageCardProps) {
  const cardId = m.id || String(index)
  const roleConf = activityRoleConfig(m.role)
  const Icon = roleConf.icon
  const isSleeptime = isSleeptimeActivity(m)
  const body = activityBodyText(m)
  const preview =
    body.length > PREVIEW_CHARS && !expanded ? body.slice(0, PREVIEW_CHARS) + '...' : body
  const canExpand = body.length > PREVIEW_CHARS

  const metaItems: string[] = []
  if (m.message_type && m.message_type !== m.role) {
    metaItems.push(m.message_type.replace(/_/g, ' '))
  }
  if (m.tool_call_id) metaItems.push(`call ${shortId(m.tool_call_id)}`)
  if (m.tool_status) metaItems.push(m.tool_status)
  if (m.step_id) metaItems.push(`step ${shortId(m.step_id)}`)
  if (m.run_id) metaItems.push(`run ${shortId(m.run_id)}`)
  if (m.sender_id) metaItems.push(`sender ${shortId(m.sender_id)}`)
  if (m.name) metaItems.push(m.name)

  return (
    <div className={`stat-card ${isSleeptime ? 'border-accent/20' : ''}`}>
      <div className="flex items-start gap-2">
        <div className={`mt-0.5 shrink-0 ${roleConf.iconClass}`}>
          <Icon className="w-3.5 h-3.5" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-1.5 mb-1">
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${roleConf.badgeClass}`}
            >
              {roleConf.label}
            </span>
            {isSleeptime && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent font-medium inline-flex items-center gap-0.5">
                <Moon className="w-2.5 h-2.5" />
                Sleeptime
              </span>
            )}
            {m.is_err && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-danger/15 text-danger font-medium inline-flex items-center gap-0.5">
                <AlertCircle className="w-2.5 h-2.5" />
                Error
              </span>
            )}
            {m.tool_status === 'error' && !m.is_err && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-danger/15 text-danger font-medium">
                Tool failed
              </span>
            )}
            {m.approve != null && (
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                  m.approve ? 'bg-success/15 text-success' : 'bg-danger/15 text-danger'
                }`}
              >
                {m.approve ? 'Approved' : 'Denied'}
              </span>
            )}
            {m.event_type && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-200 text-text-muted font-medium">
                {m.event_type}
              </span>
            )}
            {m.tool_calls?.map((tc, i) => (
              <span
                key={`${tc.name}-${i}`}
                className="text-[10px] px-1.5 py-0.5 rounded bg-warning/10 text-warning font-medium font-mono"
              >
                {tc.name}
              </span>
            ))}
            {m.created_at && (
              <span className="text-[10px] text-text-muted ml-auto shrink-0 flex items-center gap-1">
                <Clock className="w-2.5 h-2.5" />
                {new Date(m.created_at).toLocaleString()}
              </span>
            )}
          </div>

          {body ? (
            <pre className="text-xs text-text-secondary whitespace-pre-wrap font-mono leading-relaxed break-words">
              {preview}
            </pre>
          ) : (
            <p className="text-xs text-text-muted italic">No text content</p>
          )}

          {canExpand && (
            <button
              type="button"
              onClick={onToggleExpand}
              className="text-[11px] text-accent hover:text-accent-bright mt-1 flex items-center gap-1"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {expanded ? 'Less' : 'More'}
            </button>
          )}

          {m.approval_reason && (
            <p className="text-[11px] text-text-muted mt-2">
              Reason: {m.approval_reason}
            </p>
          )}

          {metaItems.length > 0 && (
            <p className="text-[10px] text-text-muted mt-2 font-mono break-all">
              {metaItems.join(' · ')}
            </p>
          )}
        </div>

        {body && (
          <button
            type="button"
            onClick={() => onCopy(body, cardId)}
            className="btn btn-ghost p-1.5 shrink-0"
            title="Copy content"
          >
            {copiedId === cardId ? (
              <Check className="w-3.5 h-3.5 text-success" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
        )}
      </div>
    </div>
  )
}

function shortId(id: string): string {
  if (id.length <= 12) return id
  return id.slice(0, 8) + '…'
}
