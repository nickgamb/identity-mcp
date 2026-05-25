import type { LucideIcon } from 'lucide-react'
import {
  Bot,
  User as UserIcon,
  Wrench,
  MessageSquare,
  Brain,
  Moon,
  AlertCircle,
  FileText,
  Zap,
  Shield,
} from 'lucide-react'

/** Normalized Letta message for Activity tab (mirrors server lettaProxy mapping). */
export interface ActivityMessage {
  id: string
  role: string
  message_type: string
  content: string | null
  created_at?: string
  tool_calls?: Array<{ name: string; arguments: string; tool_call_id?: string }>
  tool_call_id?: string
  tool_status?: 'success' | 'error'
  reasoning?: string
  summary?: string
  event_type?: string
  name?: string
  sender_id?: string
  step_id?: string
  run_id?: string
  is_err?: boolean
  approve?: boolean
  approval_reason?: string
}

export const PREVIEW_CHARS = 200

export const ACTIVITY_ROLE_CONFIG: Record<
  string,
  { label: string; icon: LucideIcon; badgeClass: string; iconClass: string }
> = {
  user: {
    label: 'User',
    icon: UserIcon,
    badgeClass: 'bg-accent/15 text-accent',
    iconClass: 'text-accent',
  },
  assistant: {
    label: 'Assistant',
    icon: Bot,
    badgeClass: 'bg-success/15 text-success',
    iconClass: 'text-success',
  },
  tool: {
    label: 'Tool result',
    icon: Wrench,
    badgeClass: 'bg-warning/15 text-warning',
    iconClass: 'text-warning',
  },
  tool_call: {
    label: 'Tool call',
    icon: Zap,
    badgeClass: 'bg-warning/15 text-warning',
    iconClass: 'text-warning',
  },
  reasoning: {
    label: 'Reasoning',
    icon: Brain,
    badgeClass: 'bg-purple-500/15 text-purple-400',
    iconClass: 'text-purple-400',
  },
  system: {
    label: 'System',
    icon: MessageSquare,
    badgeClass: 'bg-surface-200 text-text-muted',
    iconClass: 'text-text-muted',
  },
  summary: {
    label: 'Summary',
    icon: FileText,
    badgeClass: 'bg-amber-500/15 text-amber-400',
    iconClass: 'text-amber-400',
  },
  event: {
    label: 'Event',
    icon: Zap,
    badgeClass: 'bg-surface-200 text-text-secondary',
    iconClass: 'text-text-secondary',
  },
  approval: {
    label: 'Approval',
    icon: Shield,
    badgeClass: 'bg-blue-500/15 text-blue-400',
    iconClass: 'text-blue-400',
  },
}

export function activityRoleConfig(role: string) {
  return (
    ACTIVITY_ROLE_CONFIG[role] ?? {
      label: role.replace(/_/g, ' '),
      icon: MessageSquare,
      badgeClass: 'bg-surface-200 text-text-muted',
      iconClass: 'text-text-muted',
    }
  )
}

export function formatToolArguments(raw: string): string {
  const t = raw.trim()
  if (!t) return '(empty)'
  try {
    return JSON.stringify(JSON.parse(t), null, 2)
  } catch {
    return t
  }
}

/** Primary expandable body text for any activity card. */
export function activityBodyText(m: ActivityMessage): string {
  const parts: string[] = []
  if (m.content?.trim()) parts.push(m.content.trim())
  if (m.reasoning?.trim() && m.reasoning !== m.content) parts.push(m.reasoning.trim())
  if (m.summary?.trim() && m.summary !== m.content) parts.push(m.summary.trim())
  if (m.tool_calls?.length) {
    for (const tc of m.tool_calls) {
      parts.push(
        `Tool: ${tc.name}${tc.tool_call_id ? ` (${tc.tool_call_id})` : ''}\n${formatToolArguments(tc.arguments)}`
      )
    }
  }
  return parts.join('\n\n')
}

export function isSleeptimeActivity(m: ActivityMessage): boolean {
  const blob = [
    m.content,
    m.reasoning,
    m.summary,
    ...(m.tool_calls?.map(tc => `${tc.name} ${tc.arguments}`) ?? []),
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()
  if (
    blob.includes('memory') ||
    blob.includes('sleeptime') ||
    blob.includes('memory_insert') ||
    blob.includes('memory_finish') ||
    blob.includes('archival_memory') ||
    blob.includes('core_memory')
  ) {
    return true
  }
  return (
    m.tool_calls?.some(
      tc =>
        tc.name.includes('memory') ||
        tc.name.includes('archival') ||
        tc.name.includes('core_memory')
    ) ?? false
  )
}

export function activityMatchesFilter(
  m: ActivityMessage,
  filter: 'all' | 'sleeptime' | 'tools'
): boolean {
  if (filter === 'all') return true
  if (filter === 'tools') {
    return (
      m.role === 'tool' ||
      m.role === 'tool_call' ||
      (m.tool_calls?.length ?? 0) > 0 ||
      m.message_type.includes('tool')
    )
  }
  return isSleeptimeActivity(m)
}
