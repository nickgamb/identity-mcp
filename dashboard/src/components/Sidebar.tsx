import { useState, useEffect } from 'react'
import {
  Terminal,
  LayoutDashboard,
  Brain,
  MessageSquare,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
} from 'lucide-react'

export type MainView = 'pipeline' | 'data' | 'memory' | 'chat'

interface NavItem {
  id: MainView
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const NAV_ITEMS: NavItem[] = [
  { id: 'pipeline', label: 'Pipeline', icon: Terminal },
  { id: 'data', label: 'Data Explorer', icon: LayoutDashboard },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'chat', label: 'Chat', icon: MessageSquare },
]

interface SidebarProps {
  activeView: MainView
  onViewChange: (view: MainView) => void
  mcpStatus: 'checking' | 'online' | 'offline'
  lettaStatus: 'checking' | 'online' | 'offline'
  onRefreshStatus?: () => void
  hasRunningScripts?: boolean
}

export function Sidebar({
  activeView,
  onViewChange,
  mcpStatus,
  lettaStatus,
  onRefreshStatus,
  hasRunningScripts,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem('sidebar-collapsed') === 'true'
    } catch {
      return false
    }
  })

  // Persist collapse state
  useEffect(() => {
    try {
      localStorage.setItem('sidebar-collapsed', String(collapsed))
    } catch {
      // Ignore
    }
  }, [collapsed])

  // Keyboard shortcut: [ to toggle
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === '[' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const tag = (e.target as HTMLElement)?.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return
        setCollapsed(c => !c)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

  const statusDot = (status: 'checking' | 'online' | 'offline') =>
    status === 'online'
      ? 'bg-success'
      : status === 'offline'
      ? 'bg-danger'
      : 'bg-warning animate-pulse'

  return (
    <aside
      className={`flex flex-col bg-surface-50 border-r border-surface-200 transition-all duration-200 shrink-0 overflow-hidden ${
        collapsed ? 'w-sidebar-collapsed' : 'w-sidebar'
      }`}
    >
      {/* Navigation */}
      <nav className="flex-1 px-2 py-3 space-y-1">
        {NAV_ITEMS.map(item => {
          const Icon = item.icon
          const isActive = activeView === item.id
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`nav-item w-full ${
                isActive ? 'nav-item-active' : 'nav-item-inactive'
              } ${collapsed ? 'justify-center px-2' : ''}`}
              title={collapsed ? item.label : undefined}
            >
              <Icon className={`w-5 h-5 shrink-0 ${isActive ? 'text-accent' : ''}`} />
              {!collapsed && (
                <span className="truncate">{item.label}</span>
              )}
              {!collapsed && item.id === 'pipeline' && hasRunningScripts && (
                <span className="ml-auto w-2 h-2 rounded-full bg-accent animate-pulse" />
              )}
            </button>
          )
        })}
      </nav>

      {/* Status + collapse */}
      <div className="px-3 py-3 border-t border-surface-200 space-y-2">
        {/* Status indicators */}
        <div className={`flex ${collapsed ? 'flex-col items-center gap-2' : 'items-center gap-3'}`}>
          {!collapsed ? (
            <>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${statusDot(mcpStatus)}`} />
                <span className="text-[11px] text-text-muted">MCP</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${statusDot(lettaStatus)}`} />
                <span className="text-[11px] text-text-muted">Letta</span>
              </div>
              {onRefreshStatus && (
                <button
                  onClick={onRefreshStatus}
                  className="ml-auto p-1 rounded text-text-muted hover:text-text-primary hover:bg-surface-100 transition-colors"
                  title="Refresh status"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              )}
            </>
          ) : (
            <>
              <div className={`w-2 h-2 rounded-full ${statusDot(mcpStatus)}`} title={`MCP: ${mcpStatus}`} />
              <div className={`w-2 h-2 rounded-full ${statusDot(lettaStatus)}`} title={`Letta: ${lettaStatus}`} />
              {onRefreshStatus && (
                <button
                  onClick={onRefreshStatus}
                  className="p-1 rounded text-text-muted hover:text-text-primary hover:bg-surface-100 transition-colors"
                  title="Refresh status"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              )}
            </>
          )}
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(c => !c)}
          className={`nav-item nav-item-inactive w-full ${collapsed ? 'justify-center px-2' : ''}`}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span className="text-xs">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  )
}
