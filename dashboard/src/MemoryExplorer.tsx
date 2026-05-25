import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import {
  Brain,
  Database,
  Activity,
  Settings,
  RefreshCw,
  Save,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Search,
  Wrench,
  Moon,
  Clock,
  Bot,
  Cpu,
  AlertCircle,
  CheckCircle,
  Workflow,
  ArrowUpDown,
  Filter,
  Sparkles,
  Pencil,
  X,
  RotateCcw,
} from 'lucide-react'
import { authenticatedFetch } from './utils/api'
import { CodeEditor } from './components/CodeEditor'
import {
  classifyArchivalPassage,
  passageDateKey,
  passageMatchesDateRange,
  type ArchivalPassageType,
} from './utils/archivalPassage'
import { EmptyState } from './components/EmptyState'
import { MemoryMaintenance } from './components/MemoryMaintenance'
import { ActivityMessageCard } from './components/ActivityMessageCard'
import {
  type ActivityMessage,
  type ActivityFilterType,
  activityMatchesFilter,
  activityMessageKey,
  activityTimestamp,
} from './utils/lettaActivity'

// ── Ollama / Letta model handles (Letta expects ollama/model-name) ───────

function toOllamaHandle(nameOrHandle: string): string {
  const t = nameOrHandle.trim()
  if (!t) return t
  return t.startsWith('ollama/') ? t : `ollama/${t}`
}

function labelForHandle(handle: string): string {
  return handle.replace(/^ollama\//, '')
}

function buildHandleOptions(ollamaNames: string[], currentHandle: string): string[] {
  const opts = new Set<string>()
  for (const n of ollamaNames) opts.add(toOllamaHandle(n))
  if (currentHandle) opts.add(toOllamaHandle(currentHandle))
  return [...opts].sort((a, b) => labelForHandle(a).localeCompare(labelForHandle(b)))
}

// ── Types ───────────────────────────────────────────────────────────────

type Tab = 'overview' | 'maintenance' | 'core' | 'archival' | 'activity' | 'settings'

/** Map stored Letta frequency to editor (0 from API = default cadence 5). */
function sleeptimeFreqForEditor(stored: number): number {
  return stored > 0 ? stored : 5
}

interface LettaStatus {
  available: boolean
  agent?: {
    id: string
    name: string
    model: string
    embedding_model: string
    model_handle?: string
    embedding_handle?: string
    created_at: string
    enable_sleeptime?: boolean
    sleeptime_agent_frequency: number
    tool_count: number
    tools: string[]
  }
  memory?: {
    blocks: Array<{ label: string; char_count: number; limit: number }>
  }
  archival_count?: number
  archival_count_loading?: boolean
  error?: string
}

interface MemoryBlock {
  id: string
  label: string
  value: string
  limit: number
  created_at?: string
  updated_at?: string
}

interface Passage {
  id: string
  text: string
  created_at?: string
  metadata?: Record<string, any>
}

// ── Archival passage type helpers ────────────────────────────────────────

const ARCHIVAL_TYPE_CONFIG: Record<ArchivalPassageType, { label: string; className: string }> = {
  conversation: { label: 'Conversation', className: 'bg-blue-500/15 text-blue-400' },
  file:         { label: 'File',         className: 'bg-emerald-500/15 text-emerald-400' },
  memory:       { label: 'Memory',       className: 'bg-purple-500/15 text-purple-400' },
  analysis:     { label: 'Analysis',     className: 'bg-amber-500/15 text-amber-400' },
  other:        { label: 'Other',        className: 'bg-surface-200 text-text-muted' },
}

// ── Component ───────────────────────────────────────────────────────────

export function MemoryExplorer() {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [status, setStatus] = useState<LettaStatus | null>(null)
  const [loading, setLoading] = useState(true)

  // Core memory
  const [blocks, setBlocks] = useState<MemoryBlock[]>([])
  const [editingBlocks, setEditingBlocks] = useState<Record<string, string>>({})
  const [savingBlock, setSavingBlock] = useState<string | null>(null)
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null)

  // Archival
  const [passages, setPassages] = useState<Passage[]>([])
  const [archivalCursor, setArchivalCursor] = useState<string | undefined>()
  const [archivalHasMore, setArchivalHasMore] = useState(true)
  const [archivalLoading, setArchivalLoading] = useState(false)
  const [archivalSort, setArchivalSort] = useState<'oldest' | 'newest'>('newest')
  const [archivalTypeFilter, setArchivalTypeFilter] = useState<ArchivalPassageType | 'all'>('all')
  const [archivalDateFrom, setArchivalDateFrom] = useState('')
  const [archivalDateTo, setArchivalDateTo] = useState('')
  const [archivalSearch, setArchivalSearch] = useState('')
  const [searchResults, setSearchResults] = useState<Passage[] | null>(null)
  const [searchLoading, setSearchLoading] = useState(false)

  // Activity
  const [messages, setMessages] = useState<ActivityMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [activityFilter, setActivityFilter] = useState<ActivityFilterType>('all')
  const [activitySort, setActivitySort] = useState<'newest' | 'oldest'>('newest')
  const [activityDateFrom, setActivityDateFrom] = useState('')
  const [activityDateTo, setActivityDateTo] = useState('')

  // Settings
  const [sleeptimeEnabled, setSleeptimeEnabled] = useState(false)
  const [sleeptimeFreq, setSleeptimeFreq] = useState(5)
  /** Blocks status poll/tab refresh from resetting settings inputs while user is editing */
  const settingsDraftActiveRef = useRef(false)
  const markSettingsDraft = () => {
    settingsDraftActiveRef.current = true
  }
  const clearSettingsDraft = () => {
    settingsDraftActiveRef.current = false
  }
  const [updatingConfig, setUpdatingConfig] = useState(false)

  // Reverie
  const [reverieEnabled, setReverieEnabled] = useState(false)
  const [reverieInterval, setReverieInterval] = useState(120)
  const [reverieStatus, setReverieStatus] = useState<{
    config: { enabled: boolean; intervalMinutes: number }
    running: boolean
    lastReverieTime: string | null
    nextPromptLabel: string
  } | null>(null)
  const [updatingReverie, setUpdatingReverie] = useState(false)

  // Reverie prompts editor
  const [promptsEditorOpen, setPromptsEditorOpen] = useState(false)
  const [promptsJson, setPromptsJson] = useState('')
  const [promptsLoading, setPromptsLoading] = useState(false)
  const [promptsSaving, setPromptsSaving] = useState(false)
  const [promptsSaveSuccess, setPromptsSaveSuccess] = useState(false)

  const [ollamaModels, setOllamaModels] = useState<string[]>([])
  const [ollamaModelsLoading, setOllamaModelsLoading] = useState(false)
  const [ollamaModelsError, setOllamaModelsError] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedEmbedding, setSelectedEmbedding] = useState('')
  const [updatingModels, setUpdatingModels] = useState(false)
  /** Shown on Update models while Letta + Ollama unload/load run (can take several minutes). */
  const [modelUpdatePhase, setModelUpdatePhase] = useState<string | null>(null)

  // Expanded items
  const [expandedPassages, setExpandedPassages] = useState<Set<string>>(new Set())
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set())

  // Copy feedback
  const [copiedId, setCopiedId] = useState<string | null>(null)

  // ── Data loading ────────────────────────────────────────────────────

  const loadStatus = useCallback(async () => {
    try {
      const res = await authenticatedFetch('/api/mcp/letta.status')
      const data = await res.json()
      setStatus(data)
      if (data.agent && !settingsDraftActiveRef.current) {
        setSleeptimeEnabled(data.agent.enable_sleeptime ?? false)
        const freq = data.agent.sleeptime_agent_frequency ?? 0
        setSleeptimeFreq(sleeptimeFreqForEditor(freq))
        const modelHandle =
          data.agent.model_handle ||
          (data.agent.model ? toOllamaHandle(data.agent.model) : '')
        const embedHandle =
          data.agent.embedding_handle ||
          (data.agent.embedding_model ? toOllamaHandle(data.agent.embedding_model) : '')
        setSelectedModel(modelHandle)
        setSelectedEmbedding(embedHandle)
      }
    } catch (error) {
      console.error('Failed to load Letta status:', error)
      setStatus({ available: false, error: String(error) })
    }
  }, [])

  const loadCoreMemory = useCallback(async () => {
    try {
      const res = await authenticatedFetch('/api/mcp/letta.memory')
      const data = await res.json()
      if (data.blocks) {
        setBlocks(data.blocks)
        // Initialize editing state for each block
        const edits: Record<string, string> = {}
        data.blocks.forEach((b: MemoryBlock) => {
          edits[b.label] = b.value
        })
        setEditingBlocks(edits)
      }
    } catch (error) {
      console.error('Failed to load core memory:', error)
    }
  }, [])

  const loadArchival = useCallback(async (reset = false) => {
    setArchivalLoading(true)
    try {
      const params = new URLSearchParams({ limit: '50', sort: archivalSort })
      if (!reset && archivalCursor) params.set('cursor', archivalCursor)
      if (archivalTypeFilter !== 'all') params.set('type', archivalTypeFilter)
      if (archivalDateFrom) params.set('dateFrom', archivalDateFrom)
      if (archivalDateTo) params.set('dateTo', archivalDateTo)
      const res = await authenticatedFetch(`/api/mcp/letta.archival?${params}`)
      const data = await res.json()
      if (data.passages) {
        if (reset) {
          setPassages(data.passages)
        } else {
          setPassages(prev => [...prev, ...data.passages])
        }
        setArchivalCursor(data.nextCursor)
        setArchivalHasMore(data.hasMore ?? data.passages.length >= 50)
      }
    } catch (error) {
      console.error('Failed to load archival:', error)
    } finally {
      setArchivalLoading(false)
    }
  }, [archivalCursor, archivalSort, archivalTypeFilter, archivalDateFrom, archivalDateTo])

  const searchArchival = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSearchResults(null)
      return
    }
    setSearchLoading(true)
    try {
      const res = await authenticatedFetch('/api/mcp/search.semantic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, limit: 30 }),
      })
      const data = await res.json()
      const results = (data.results || []).map((r: any) => ({
        id: r.id || crypto.randomUUID(),
        text: r.text || '',
        created_at: r.created_at,
        metadata: {},
      }))
      setSearchResults(results)
    } catch (error) {
      console.error('Archival search failed:', error)
    } finally {
      setSearchLoading(false)
    }
  }, [])

  const loadOllamaModels = useCallback(async () => {
    setOllamaModelsLoading(true)
    setOllamaModelsError(null)
    try {
      const res = await authenticatedFetch('/api/mcp/ollama.models')
      const data = await res.json()
      if (data.available && Array.isArray(data.models)) {
        setOllamaModels(data.models)
      } else {
        setOllamaModels([])
        setOllamaModelsError(data.error || 'Could not list Ollama models')
      }
    } catch (error) {
      setOllamaModels([])
      setOllamaModelsError(String(error))
    } finally {
      setOllamaModelsLoading(false)
    }
  }, [])

  const loadReverieStatus = useCallback(async () => {
    try {
      const res = await authenticatedFetch('/api/mcp/reverie.status')
      const data = await res.json()
      setReverieStatus(data)
      if (!settingsDraftActiveRef.current) {
        setReverieEnabled(data.config?.enabled ?? false)
        setReverieInterval(data.config?.intervalMinutes ?? 120)
      }
    } catch (error) {
      console.error('Failed to load reverie status:', error)
    }
  }, [])

  const loadMessages = useCallback(async () => {
    setMessagesLoading(true)
    try {
      const res = await authenticatedFetch('/api/mcp/letta.messages?limit=100')
      const data = await res.json()
      if (data.messages) {
        setMessages(data.messages)
      }
    } catch (error) {
      console.error('Failed to load messages:', error)
    } finally {
      setMessagesLoading(false)
    }
  }, [])

  // ── Initial load ────────────────────────────────────────────────────

  useEffect(() => {
    const init = async () => {
      await loadStatus()
      setLoading(false)
    }
    init()

    const interval = setInterval(loadStatus, 30000)
    return () => clearInterval(interval)
  }, [loadStatus])

  // Poll while archival count warms up in background (after server restart / cold cache)
  useEffect(() => {
    if (!status?.available || status.archival_count != null) return
    const poll = setInterval(loadStatus, 4000)
    return () => clearInterval(poll)
  }, [status?.available, status?.archival_count, loadStatus])

  useEffect(() => {
    if (activeTab === 'core') loadCoreMemory()
    if (activeTab === 'activity') loadMessages()
    if (activeTab === 'settings') {
      loadStatus()
      loadOllamaModels()
      loadReverieStatus()
    }
  }, [activeTab, loadCoreMemory, loadMessages, loadOllamaModels, loadReverieStatus, loadStatus])

  // Archival browse: reload when tab, sort, type, or date range changes (server-side scan)
  useEffect(() => {
    if (activeTab !== 'archival' || searchResults) return
    setPassages([])
    setArchivalCursor(undefined)
    setArchivalHasMore(true)
    loadArchival(true)
  // eslint-disable-next-line react-hooks/exhaustive-deps -- intentional: reset list when browse params change
  }, [activeTab, archivalTypeFilter, archivalSort, archivalDateFrom, archivalDateTo, searchResults])

  // ── Actions ─────────────────────────────────────────────────────────

  const saveCoreMemory = async (label: string) => {
    setSavingBlock(label)
    try {
      const res = await authenticatedFetch('/api/mcp/letta.memory.update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blockLabel: label, value: editingBlocks[label] }),
      })
      const data = await res.json()
      if (data.success) {
        setSaveSuccess(label)
        setTimeout(() => setSaveSuccess(null), 2000)
        // Refresh blocks
        await loadCoreMemory()
      } else {
        alert(`Failed to save: ${data.error}`)
      }
    } catch (error) {
      alert(`Save error: ${error}`)
    } finally {
      setSavingBlock(null)
    }
  }

  const updateSleeptimeConfig = async () => {
    setUpdatingConfig(true)
    try {
      const patch: Record<string, unknown> = {
        enable_sleeptime: sleeptimeEnabled,
      }
      if (sleeptimeEnabled) {
        patch.sleeptime_agent_frequency = Math.max(1, sleeptimeFreq)
      }
      const res = await authenticatedFetch('/api/mcp/letta.config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      const data = await res.json()
      if (!res.ok) {
        alert(`Update failed: ${data.error || res.statusText}`)
        return
      }
      if (data.success) {
        clearSettingsDraft()
        setSaveSuccess('sleeptime')
        setTimeout(() => setSaveSuccess(null), 2000)
        await loadStatus()
      } else {
        alert(`Update failed: ${data.error}`)
      }
    } catch (error) {
      alert(`Update error: ${error}`)
    } finally {
      setUpdatingConfig(false)
    }
  }

  const updateReverieConfig = async () => {
    setUpdatingReverie(true)
    try {
      const res = await authenticatedFetch('/api/mcp/reverie.config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: reverieEnabled,
          intervalMinutes: Math.max(30, reverieInterval),
        }),
      })
      const data = await res.json()
      if (data.success) {
        clearSettingsDraft()
        setSaveSuccess('reverie')
        setTimeout(() => setSaveSuccess(null), 2000)
        await loadReverieStatus()
      } else {
        alert(`Update failed: ${data.error}`)
      }
    } catch (error) {
      alert(`Update error: ${error}`)
    } finally {
      setUpdatingReverie(false)
    }
  }

  const loadReveriePrompts = useCallback(async () => {
    setPromptsLoading(true)
    try {
      const res = await authenticatedFetch('/api/mcp/reverie.prompts')
      const data = await res.json()
      setPromptsJson(JSON.stringify(data.prompts ?? [], null, 2))
    } catch (error) {
      console.error('Failed to load reverie prompts:', error)
      setPromptsJson('[]')
    } finally {
      setPromptsLoading(false)
    }
  }, [])

  const saveReveriePrompts = useCallback(async () => {
    let parsed: unknown
    try {
      parsed = JSON.parse(promptsJson)
    } catch {
      alert('Invalid JSON — please fix syntax errors before saving.')
      return
    }
    if (!Array.isArray(parsed) || parsed.length === 0) {
      alert('Prompts must be a non-empty JSON array.')
      return
    }
    for (const p of parsed as any[]) {
      if (!p.label?.trim() || !p.text?.trim()) {
        alert('Each prompt needs both a "label" and "text" field.')
        return
      }
    }
    setPromptsSaving(true)
    try {
      const res = await authenticatedFetch('/api/mcp/reverie.prompts', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompts: parsed }),
      })
      const data = await res.json()
      if (data.success) {
        setPromptsSaveSuccess(true)
        setTimeout(() => setPromptsSaveSuccess(false), 2000)
        await loadReverieStatus()
      } else {
        alert(`Save failed: ${data.error}`)
      }
    } catch (error) {
      alert(`Save error: ${error}`)
    } finally {
      setPromptsSaving(false)
    }
  }, [promptsJson, loadReverieStatus])

  const reverieConfigDirty =
    reverieStatus != null &&
    (reverieEnabled !== reverieStatus.config.enabled ||
      reverieInterval !== reverieStatus.config.intervalMinutes)

  const sleeptimeConfigDirty =
    status?.agent &&
    (sleeptimeEnabled !== (status.agent.enable_sleeptime ?? false) ||
      (sleeptimeEnabled &&
        sleeptimeFreq !==
          sleeptimeFreqForEditor(status.agent.sleeptime_agent_frequency ?? 0)))

  const savedModelHandle =
    status?.agent?.model_handle ||
    (status?.agent?.model ? toOllamaHandle(status.agent.model) : '')
  const savedEmbeddingHandle =
    status?.agent?.embedding_handle ||
    (status?.agent?.embedding_model ? toOllamaHandle(status.agent.embedding_model) : '')

  const modelsConfigDirty =
    !!status?.agent &&
    (toOllamaHandle(selectedModel) !== toOllamaHandle(savedModelHandle) ||
      toOllamaHandle(selectedEmbedding) !== toOllamaHandle(savedEmbeddingHandle))

  const modelHandleOptions = buildHandleOptions(ollamaModels, savedModelHandle)
  const embeddingHandleOptions = buildHandleOptions(ollamaModels, savedEmbeddingHandle)

  const updateAgentModels = async () => {
    setUpdatingModels(true)
    setModelUpdatePhase('Saving to Letta and loading model in Ollama (this may take several minutes)…')
    try {
      const res = await authenticatedFetch('/api/mcp/letta.config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: selectedModel,
          embedding: selectedEmbedding,
        }),
        signal: AbortSignal.timeout(20 * 60 * 1000),
      })
      const data = await res.json()
      if (data.success) {
        if (data.model_sync?.compaction_model) {
          setModelUpdatePhase(
            `Synced ${data.model_sync.compaction_model} (compaction + sleeptime agents)`,
          )
        } else if (data.ollama?.message) {
          setModelUpdatePhase(data.ollama.message)
        }
        clearSettingsDraft()
        await loadStatus()
      } else {
        alert(`Update failed: ${data.error || data.ollama?.error || 'Unknown error'}`)
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error)
      alert(
        msg.includes('timeout') || msg.includes('aborted')
          ? 'Update timed out while loading the model in Ollama. Check server GPU load and try again.'
          : `Update error: ${msg}`,
      )
    } finally {
      setUpdatingModels(false)
      setModelUpdatePhase(null)
    }
  }

  const copyToClipboard = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 1500)
  }, [])

  const toggleExpand = (id: string, set: Set<string>, setter: (s: Set<string>) => void) => {
    const next = new Set(set)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setter(next)
  }

  const toggleActivityExpand = useCallback((cardKey: string) => {
    setExpandedMessages(prev => {
      const next = new Set(prev)
      if (next.has(cardKey)) next.delete(cardKey)
      else next.add(cardKey)
      return next
    })
  }, [])

  // ── Derived ─────────────────────────────────────────────────────────

  const filteredMessages = useMemo(() => {
    const filtered = messages.filter(m => {
      if (!activityMatchesFilter(m, activityFilter)) return false
      if (activityDateFrom && m.created_at && m.created_at.slice(0, 10) < activityDateFrom) return false
      if (activityDateTo && m.created_at && m.created_at.slice(0, 10) > activityDateTo) return false
      return true
    })
    return [...filtered].sort((a, b) => {
      const diff = activityTimestamp(b) - activityTimestamp(a)
      return activitySort === 'newest' ? diff : -diff
    })
  }, [messages, activityFilter, activityDateFrom, activityDateTo, activitySort])

  // ── Tab config ──────────────────────────────────────────────────────

  const TABS: Array<{ id: Tab; label: string; icon: React.ComponentType<{ className?: string }> }> = [
    { id: 'overview', label: 'Overview', icon: Brain },
    { id: 'core', label: 'Core Memory', icon: Database },
    { id: 'archival', label: 'Archival', icon: Search },
    { id: 'activity', label: 'Activity', icon: Activity },
    { id: 'maintenance', label: 'Maintenance', icon: Workflow },
    { id: 'settings', label: 'Settings', icon: Settings },
  ]

  // ── Unavailable state ───────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4">
        <div className="w-10 h-10 border-4 border-surface-300 border-t-accent rounded-full animate-spin" />
        <p className="text-text-muted text-sm">Connecting to Letta...</p>
      </div>
    )
  }

  if (status && !status.available) {
    return (
      <div className="flex flex-col items-center justify-center py-24">
        <div className="card max-w-md text-center">
          <AlertCircle className="w-16 h-16 mx-auto mb-4 text-warning opacity-60" />
          <h2 className="font-display text-xl font-semibold text-text-primary mb-2">Letta Unavailable</h2>
          <p className="text-text-secondary mb-4">
            {status.error || 'Could not connect to the Letta memory system.'}
          </p>
          <p className="text-text-muted text-sm mb-6">
            Ensure the Letta service is running and the identity agent exists.
          </p>
          <button onClick={() => { setLoading(true); loadStatus().then(() => setLoading(false)) }} className="btn btn-primary">
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    )
  }

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div>
      {/* Tabs */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-1">
        {TABS.map(tab => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                isActive
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-secondary hover:bg-surface-100 hover:text-text-primary'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          )
        })}
        <div className="flex-1" />
        <button onClick={loadStatus} className="btn btn-ghost text-xs" title="Refresh">
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* ── Overview ─────────────────────────────────────────────── */}
      {activeTab === 'overview' && status?.agent && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {/* Agent Status */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Bot className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Agent</h3>
              <div className="ml-auto w-2 h-2 rounded-full bg-success" />
            </div>
            <p className="text-lg font-display font-semibold text-text-primary mb-1">
              {status.agent.name}
            </p>
            <p className="text-xs text-text-muted truncate" title={status.agent.model}>
              {status.agent.model}
            </p>
            <p className="text-xs text-text-muted mt-1">
              Created {status.agent.created_at ? new Date(status.agent.created_at).toLocaleDateString() : 'unknown'}
            </p>
          </div>

          {/* Core Memory */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Core Memory</h3>
            </div>
            {status.memory?.blocks.map(b => (
              <div key={b.label} className="mb-2 last:mb-0">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-text-secondary capitalize">{b.label}</span>
                  <span className="text-xs text-text-muted">
                    {b.char_count.toLocaleString()} / {b.limit.toLocaleString()}
                  </span>
                </div>
                <div className="w-full bg-surface-200 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all ${
                      b.limit > 0 && b.char_count / b.limit > 0.9
                        ? 'bg-danger'
                        : b.limit > 0 && b.char_count / b.limit > 0.7
                        ? 'bg-warning'
                        : 'bg-accent'
                    }`}
                    style={{ width: `${b.limit > 0 ? Math.min(100, (b.char_count / b.limit) * 100) : 0}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Archival Memory */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Search className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Archival Memory</h3>
            </div>
            {status.archival_count == null ? (
              <div className="flex items-center gap-3 py-3">
                <div
                  className="w-9 h-9 border-4 border-surface-300 border-t-accent rounded-full animate-spin shrink-0"
                  aria-hidden
                />
                <div>
                  <p className="text-sm font-medium text-text-primary">Counting passages…</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    {status.archival_count_loading
                      ? 'Scanning pgvector archive'
                      : 'Starting count'}
                  </p>
                </div>
              </div>
            ) : (
              <>
                <p className="text-3xl font-display font-bold text-text-primary">
                  {status.archival_count.toLocaleString()}
                </p>
                <p className="text-xs text-text-muted mt-1">passages in pgvector</p>
              </>
            )}
          </div>

          {/* Sleeptime */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Moon className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Sleeptime</h3>
            </div>
            <p className="text-2xl font-display font-bold text-text-primary">
              {!status.agent.enable_sleeptime
                ? 'Disabled'
                : status.agent.sleeptime_agent_frequency === 0
                ? 'On'
                : `Every ${status.agent.sleeptime_agent_frequency}`}
            </p>
            <p className="text-xs text-text-muted mt-1">
              {!status.agent.enable_sleeptime
                ? 'Background processing disabled'
                : status.agent.sleeptime_agent_frequency === 1
                ? 'Processes after every message'
                : status.agent.sleeptime_agent_frequency === 0
                ? 'Sleeptime enabled (default cadence)'
                : `Processes every ${status.agent.sleeptime_agent_frequency} messages`}
            </p>
          </div>

          {/* Tools */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Wrench className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Tools</h3>
            </div>
            <p className="text-2xl font-display font-bold text-text-primary mb-2">
              {status.agent.tool_count}
            </p>
            <div className="flex flex-wrap gap-1">
              {status.agent.tools.slice(0, 8).map(t => (
                <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-surface-200 text-text-muted truncate max-w-[120px]">
                  {t}
                </span>
              ))}
              {status.agent.tools.length > 8 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-200 text-text-muted">
                  +{status.agent.tools.length - 8} more
                </span>
              )}
            </div>
          </div>

          {/* Embedding Model */}
          <div className="stat-card">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-text-primary">Embedding</h3>
            </div>
            <p className="text-sm font-medium text-text-primary truncate" title={status.agent.embedding_model}>
              {status.agent.embedding_model}
            </p>
            <p className="text-xs text-text-muted mt-1">vector embedding model</p>
          </div>
        </div>
      )}

      {/* ── Maintenance ─────────────────────────────────────────── */}
      {activeTab === 'maintenance' && (
        <MemoryMaintenance onJobComplete={loadStatus} />
      )}

      {/* ── Core Memory ──────────────────────────────────────────── */}
      {activeTab === 'core' && (
        <div className="space-y-6">
          {blocks.length === 0 ? (
            <EmptyState icon={Database} title="No Blocks" message="No memory blocks loaded" />
          ) : (
            blocks.map(block => {
              const charCount = (editingBlocks[block.label] || '').length
              const pct = block.limit > 0 ? (charCount / block.limit) * 100 : 0
              const isDirty = editingBlocks[block.label] !== block.value
              return (
                <div key={block.label} className="card">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4 text-accent" />
                      <h3 className="font-display font-semibold text-text-primary capitalize">
                        {block.label}
                      </h3>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs ${pct > 90 ? 'text-danger' : pct > 70 ? 'text-warning' : 'text-text-muted'}`}>
                        {charCount.toLocaleString()} / {block.limit.toLocaleString()} chars
                      </span>
                      <button
                        onClick={() => saveCoreMemory(block.label)}
                        disabled={!isDirty || savingBlock === block.label}
                        className={`btn text-sm ${isDirty ? 'btn-primary' : 'btn-ghost opacity-50 cursor-not-allowed'}`}
                      >
                        {savingBlock === block.label ? (
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        ) : saveSuccess === block.label ? (
                          <Check className="w-3.5 h-3.5 text-success" />
                        ) : (
                          <Save className="w-3.5 h-3.5" />
                        )}
                        {savingBlock === block.label ? 'Saving' : saveSuccess === block.label ? 'Saved' : 'Save'}
                      </button>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full bg-surface-200 rounded-full h-1 mb-3">
                    <div
                      className={`h-1 rounded-full transition-all ${
                        pct > 90 ? 'bg-danger' : pct > 70 ? 'bg-warning' : 'bg-accent'
                      }`}
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>

                  {/* Editor */}
                  <textarea
                    value={editingBlocks[block.label] || ''}
                    onChange={e => setEditingBlocks(prev => ({ ...prev, [block.label]: e.target.value }))}
                    className="w-full bg-surface font-mono text-sm text-text-primary border border-surface-200 rounded-lg p-4 min-h-[200px] resize-y focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/50"
                    spellCheck={false}
                  />

                  {block.updated_at && (
                    <p className="text-xs text-text-muted mt-2">
                      Last updated: {new Date(block.updated_at).toLocaleString()}
                    </p>
                  )}
                </div>
              )
            })
          )}
        </div>
      )}

      {/* ── Archival Memory ──────────────────────────────────────── */}
      {activeTab === 'archival' && (() => {
        const allPassages = searchResults || passages
        const hasDateFilters = !!(archivalDateFrom || archivalDateTo)
        // Semantic search results: apply date filter client-side (header date preferred)
        const filteredPassages = searchResults
          ? allPassages.filter(p =>
              passageMatchesDateRange(p.text, p.created_at, archivalDateFrom, archivalDateTo)
            )
          : allPassages
        const typeFilterActive = archivalTypeFilter !== 'all' && !searchResults

        return (
        <div>
          {/* Search + Sort */}
          <div className="flex items-center gap-2 mb-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-muted" />
              <input
                type="text"
                placeholder="Semantic search across archival memory..."
                value={archivalSearch}
                onChange={e => setArchivalSearch(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') searchArchival(archivalSearch) }}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 bg-surface-50 focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none"
              />
            </div>
            <button
              onClick={() => {
                setArchivalSort(archivalSort === 'newest' ? 'oldest' : 'newest')
              }}
              className="btn btn-ghost text-xs shrink-0 gap-1.5"
              title={`Sorted ${archivalSort} first — click to flip`}
            >
              <ArrowUpDown className="w-3.5 h-3.5" />
              {archivalSort === 'newest' ? 'Newest' : 'Oldest'}
            </button>
          </div>

          {/* Filters: type pills + date range */}
          <div className="flex flex-wrap items-center gap-2 mb-4">
            {(['all', 'conversation', 'file', 'memory', 'analysis', 'other'] as const).map(f => (
              <button
                key={f}
                onClick={() => setArchivalTypeFilter(f)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  archivalTypeFilter === f
                    ? 'bg-accent/15 text-accent'
                    : 'bg-surface-100 text-text-secondary hover:bg-surface-200'
                }`}
              >
                {f === 'all' ? 'All' : ARCHIVAL_TYPE_CONFIG[f].label}
              </button>
            ))}
            <div className="flex-1" />
            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <span className="font-medium text-text-secondary">Date:</span>
              <input
                type="date"
                value={archivalDateFrom}
                onChange={e => setArchivalDateFrom(e.target.value)}
                className="px-2 py-1 rounded border border-surface-200 bg-surface-50 text-text-secondary text-xs outline-none focus:border-accent"
                title="From date"
              />
              <span>–</span>
              <input
                type="date"
                value={archivalDateTo}
                onChange={e => setArchivalDateTo(e.target.value)}
                className="px-2 py-1 rounded border border-surface-200 bg-surface-50 text-text-secondary text-xs outline-none focus:border-accent"
                title="To date"
              />
              {hasDateFilters && (
                <button
                  onClick={() => { setArchivalDateFrom(''); setArchivalDateTo('') }}
                  className="btn btn-ghost text-xs px-2 py-1 text-accent"
                  title="Clear all filters"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Results */}
          {(searchLoading || (archivalLoading && passages.length === 0)) ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-accent" />
            </div>
          ) : allPassages.length === 0 ? (
            <EmptyState icon={Search} title={searchResults ? 'No Results' : 'No Passages'} message={searchResults ? 'No results found for your query' : 'No archival passages loaded yet'} />
          ) : filteredPassages.length === 0 ? (
            <EmptyState
              icon={Filter}
              title="No Matches"
              message={
                typeFilterActive
                  ? `No ${ARCHIVAL_TYPE_CONFIG[archivalTypeFilter].label.toLowerCase()} passages found in this scan window`
                  : `No passages match the date filters (${allPassages.length} loaded)`
              }
            />
          ) : (
            <>
              <div className="flex items-center justify-between mb-3">
                <p className="text-sm text-text-muted">
                  {searchResults
                    ? `${filteredPassages.length}${hasDateFilters ? ` of ${searchResults.length}` : ''} search results`
                    : typeFilterActive
                      ? `${filteredPassages.length} ${ARCHIVAL_TYPE_CONFIG[archivalTypeFilter].label.toLowerCase()} passages${archivalHasMore ? ' (load more to scan)' : ''}`
                      : `${filteredPassages.length}${hasDateFilters ? ` of ${passages.length} loaded` : ''} passages${status?.archival_count ? ` (${status.archival_count.toLocaleString()} total)` : ''}`}
                </p>
                {searchResults && (
                  <button
                    onClick={() => { setSearchResults(null); setArchivalSearch('') }}
                    className="btn btn-ghost text-xs"
                  >
                    Clear search
                  </button>
                )}
              </div>

              <div className="space-y-2">
                {filteredPassages.map((p, idx) => {
                  const isExpanded = expandedPassages.has(p.id || String(idx))
                  const preview = p.text.length > 200 && !isExpanded ? p.text.slice(0, 200) + '...' : p.text
                  const pType = classifyArchivalPassage(p.text)
                  const displayDate = passageDateKey(p.text, p.created_at)
                  const typeConf = ARCHIVAL_TYPE_CONFIG[pType]
                  return (
                    <div key={p.id || idx} className="stat-card">
                      <div className="flex items-start gap-3">
                        <div className="flex-1 min-w-0">
                          <pre className="text-sm text-text-secondary whitespace-pre-wrap font-mono leading-relaxed break-words">
                            {preview}
                          </pre>
                          {p.text.length > 200 && (
                            <button
                              onClick={() => toggleExpand(p.id || String(idx), expandedPassages, setExpandedPassages)}
                              className="text-xs text-accent hover:text-accent-bright mt-1 flex items-center gap-1"
                            >
                              {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                              {isExpanded ? 'Show less' : 'Show more'}
                            </button>
                          )}
                        </div>
                        <button
                          onClick={() => copyToClipboard(p.text, p.id || String(idx))}
                          className="btn btn-ghost p-1.5 shrink-0"
                          title="Copy text"
                        >
                          {copiedId === (p.id || String(idx)) ? (
                            <Check className="w-3.5 h-3.5 text-success" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium ${typeConf.className}`}>
                          {typeConf.label}
                        </span>
                        {displayDate && (
                          <>
                            <Clock className="w-3 h-3 text-text-muted" />
                            <span className="text-[11px] text-text-muted" title={p.created_at || undefined}>
                              {displayDate}
                              {p.created_at && passageDateKey(p.text, null) === null
                                ? ` · ${new Date(p.created_at).toLocaleString()}`
                                : ''}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Load more (browse mode only) */}
              {!searchResults && archivalHasMore && (
                <div className="flex justify-center mt-4">
                  <button
                    onClick={() => loadArchival(false)}
                    disabled={archivalLoading}
                    className="btn btn-ghost"
                  >
                    {archivalLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Loading...
                      </>
                    ) : (
                      typeFilterActive ? 'Scan more' : 'Load more'
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
        )
      })()}

      {/* ── Activity ─────────────────────────────────────────────── */}
      {activeTab === 'activity' && (
        <div>
          {/* Filter */}
          <div className="flex items-center gap-2 mb-4">
            {(['all', 'sleeptime', 'reverie', 'tools'] as const).map(f => (
              <button
                key={f}
                onClick={() => setActivityFilter(f)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  activityFilter === f
                    ? 'bg-accent/15 text-accent'
                    : 'bg-surface-100 text-text-secondary hover:bg-surface-200'
                }`}
              >
                {f === 'all' ? 'All' : f === 'sleeptime' ? 'Sleeptime' : f === 'reverie' ? 'Reverie' : 'Tool Calls'}
              </button>
            ))}
            <div className="flex-1" />
            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <span className="font-medium text-text-secondary">Date:</span>
              <input
                type="date"
                value={activityDateFrom}
                onChange={e => setActivityDateFrom(e.target.value)}
                className="px-2 py-1 rounded border border-surface-200 bg-surface-50 text-text-secondary text-xs outline-none focus:border-accent"
                title="From date"
              />
              <span>–</span>
              <input
                type="date"
                value={activityDateTo}
                onChange={e => setActivityDateTo(e.target.value)}
                className="px-2 py-1 rounded border border-surface-200 bg-surface-50 text-text-secondary text-xs outline-none focus:border-accent"
                title="To date"
              />
              {(activityDateFrom || activityDateTo) && (
                <button
                  onClick={() => { setActivityDateFrom(''); setActivityDateTo('') }}
                  className="btn btn-ghost text-xs px-2 py-1 text-accent"
                  title="Clear date filter"
                >
                  Clear
                </button>
              )}
            </div>
            <button
              onClick={() => setActivitySort(prev => prev === 'newest' ? 'oldest' : 'newest')}
              className="btn btn-ghost text-xs shrink-0 gap-1.5"
              title={`Sorted ${activitySort} first — click to flip`}
            >
              <ArrowUpDown className="w-3.5 h-3.5" />
              {activitySort === 'newest' ? 'Newest' : 'Oldest'}
            </button>
            <button onClick={loadMessages} disabled={messagesLoading} className="btn btn-ghost text-xs">
              <RefreshCw className={`w-3.5 h-3.5 ${messagesLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>

          {messagesLoading && messages.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-accent" />
            </div>
          ) : filteredMessages.length === 0 ? (
            <EmptyState icon={Activity} title="No Activity" message={activityFilter === 'all' ? 'No messages yet' : `No ${activityFilter === 'tools' ? 'tool call' : activityFilter} activity found`} />
          ) : (
            <div className="space-y-1.5">
              {filteredMessages.map((m, idx) => {
                const cardKey = activityMessageKey(m, idx)
                return (
                  <ActivityMessageCard
                    key={cardKey}
                    message={m}
                    cardKey={cardKey}
                    expanded={expandedMessages.has(cardKey)}
                    copiedId={copiedId}
                    onToggleExpand={toggleActivityExpand}
                    onCopy={copyToClipboard}
                  />
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* ── Settings ─────────────────────────────────────────────── */}
      {activeTab === 'settings' && status?.agent && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Sleeptime */}
          <div className="card lg:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Moon className="w-5 h-5 text-accent" />
              <h3 className="font-display font-semibold text-text-primary">Sleeptime</h3>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              Background memory consolidation runs on a separate sleeptime agent. Enable it to evolve core memory from conversations; set frequency to 1 for every message or higher to reduce load.
            </p>

            <div className="flex items-center justify-between gap-4 mb-5 py-3 px-3 rounded-lg bg-surface-100 border border-surface-200">
              <div>
                <p className="text-sm font-medium text-text-primary">Enable sleeptime</p>
                <p className="text-xs text-text-muted mt-0.5">
                  {sleeptimeEnabled ? 'Background processing active' : 'Paused — no overnight consolidation'}
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={sleeptimeEnabled}
                data-on={sleeptimeEnabled}
                className="toggle-track"
                onClick={() => {
                  markSettingsDraft()
                  setSleeptimeEnabled(v => !v)
                }}
              >
                <span className="toggle-thumb" />
              </button>
            </div>

            <div className={`flex flex-wrap items-center gap-4 ${!sleeptimeEnabled ? 'opacity-50' : ''}`}>
              <label className="text-sm text-text-secondary">Frequency</label>
              <div className="input-number">
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={sleeptimeFreq}
                  disabled={!sleeptimeEnabled}
                  onChange={e => {
                    markSettingsDraft()
                    setSleeptimeFreq(Math.min(100, Math.max(1, parseInt(e.target.value, 10) || 1)))
                  }}
                  className="input-number-field"
                />
                <div className="input-number-step">
                  <button
                    type="button"
                    disabled={!sleeptimeEnabled || sleeptimeFreq >= 100}
                    onClick={() => {
                      markSettingsDraft()
                      setSleeptimeFreq(f => Math.min(100, f + 1))
                    }}
                    aria-label="Increase frequency"
                  >
                    <ChevronUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    disabled={!sleeptimeEnabled || sleeptimeFreq <= 1}
                    onClick={() => {
                      markSettingsDraft()
                      setSleeptimeFreq(f => Math.max(1, f - 1))
                    }}
                    aria-label="Decrease frequency"
                  >
                    <ChevronDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <span className="text-sm text-text-muted">
                {sleeptimeFreq === 1 ? 'Every message' : `Every ${sleeptimeFreq} messages`}
              </span>
              <button
                type="button"
                onClick={updateSleeptimeConfig}
                disabled={updatingConfig || !sleeptimeConfigDirty}
                className="btn btn-primary ml-auto"
                title={
                  !sleeptimeConfigDirty
                    ? 'Change frequency or toggles to enable Save'
                    : undefined
                }
              >
                {updatingConfig ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : saveSuccess === 'sleeptime' ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                {updatingConfig ? 'Saving…' : saveSuccess === 'sleeptime' ? 'Saved' : 'Save'}
              </button>
            </div>
          </div>

          {/* Reverie */}
          <div className="card lg:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-5 h-5 text-accent" />
              <h3 className="font-display font-semibold text-text-primary">Reverie</h3>
              {reverieStatus?.running && (
                <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-accent/15 text-accent animate-pulse">
                  Running
                </span>
              )}
            </div>
            <p className="text-sm text-text-secondary mb-4">
              Periodic self-reflection when the GPU is idle. The agent dream-walks its memories — reviewing conversations, noticing patterns, refining its self-model.
            </p>

            <div className="flex items-center justify-between gap-4 mb-5 py-3 px-3 rounded-lg bg-surface-100 border border-surface-200">
              <div>
                <p className="text-sm font-medium text-text-primary">Enable reverie</p>
                <p className="text-xs text-text-muted mt-0.5">
                  {reverieEnabled ? 'Will reflect when GPU is idle' : 'Disabled — no background reflection'}
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={reverieEnabled}
                data-on={reverieEnabled}
                className="toggle-track"
                onClick={() => {
                  markSettingsDraft()
                  setReverieEnabled(v => !v)
                }}
              >
                <span className="toggle-thumb" />
              </button>
            </div>

            <div className={`flex flex-wrap items-center gap-4 ${!reverieEnabled ? 'opacity-50' : ''}`}>
              <label className="text-sm text-text-secondary">Interval</label>
              <div className="input-number">
                <input
                  type="number"
                  min={30}
                  max={720}
                  value={reverieInterval}
                  disabled={!reverieEnabled}
                  onChange={e => {
                    markSettingsDraft()
                    setReverieInterval(Math.min(720, Math.max(30, parseInt(e.target.value, 10) || 30)))
                  }}
                  className="input-number-field"
                />
                <div className="input-number-step">
                  <button
                    type="button"
                    disabled={!reverieEnabled || reverieInterval >= 720}
                    onClick={() => {
                      markSettingsDraft()
                      setReverieInterval(f => Math.min(720, f + 30))
                    }}
                    aria-label="Increase interval"
                  >
                    <ChevronUp className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    disabled={!reverieEnabled || reverieInterval <= 30}
                    onClick={() => {
                      markSettingsDraft()
                      setReverieInterval(f => Math.max(30, f - 30))
                    }}
                    aria-label="Decrease interval"
                  >
                    <ChevronDown className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              <span className="text-sm text-text-muted">
                {reverieInterval < 60
                  ? `${reverieInterval} min`
                  : reverieInterval === 60
                    ? '1 hour'
                    : `${(reverieInterval / 60).toFixed(1).replace(/\.0$/, '')} hours`}
              </span>
              <button
                type="button"
                onClick={updateReverieConfig}
                disabled={updatingReverie || !reverieConfigDirty}
                className="btn btn-primary ml-auto"
                title={
                  !reverieConfigDirty
                    ? 'Change interval or toggle to enable Save'
                    : undefined
                }
              >
                {updatingReverie ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : saveSuccess === 'reverie' ? (
                  <CheckCircle className="w-4 h-4" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                {updatingReverie ? 'Saving…' : saveSuccess === 'reverie' ? 'Saved' : 'Save'}
              </button>
            </div>

            {reverieStatus?.lastReverieTime && (
              <p className="text-xs text-text-muted mt-4">
                Last reverie: {new Date(reverieStatus.lastReverieTime).toLocaleString()}
                {' — next prompt: '}{reverieStatus.nextPromptLabel}
              </p>
            )}

            <div className="mt-4 pt-4 border-t border-surface-200">
              <button
                type="button"
                className="btn btn-ghost text-sm gap-1.5 text-accent hover:bg-accent/10"
                onClick={() => {
                  loadReveriePrompts()
                  setPromptsEditorOpen(true)
                }}
              >
                <Pencil className="w-3.5 h-3.5" />
                Edit Prompts
              </button>
            </div>

            {promptsEditorOpen && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                <div className="bg-surface-50 border border-surface-200 rounded-xl shadow-xl w-[90vw] max-w-3xl max-h-[85vh] flex flex-col">
                  <div className="flex items-center justify-between px-5 py-3 border-b border-surface-200">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-accent" />
                      <h3 className="font-display font-semibold text-text-primary text-sm">Reverie Prompts</h3>
                    </div>
                    <button
                      type="button"
                      className="btn btn-ghost p-1.5"
                      onClick={() => setPromptsEditorOpen(false)}
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="flex-1 min-h-0 overflow-hidden">
                    {promptsLoading ? (
                      <div className="flex items-center justify-center h-64 text-text-muted text-sm">
                        <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading prompts…
                      </div>
                    ) : (
                      <CodeEditor
                        value={promptsJson}
                        onChange={(v) => setPromptsJson(v ?? '')}
                        language="json"
                        height="55vh"
                      />
                    )}
                  </div>

                  <div className="flex items-center justify-between px-5 py-3 border-t border-surface-200">
                    <button
                      type="button"
                      className="btn btn-ghost text-xs gap-1.5"
                      title="Reset to built-in defaults"
                      onClick={async () => {
                        setPromptsLoading(true)
                        try {
                          const mod = await import('./utils/reverieDefaults')
                          setPromptsJson(JSON.stringify(mod.DEFAULT_REVERIE_PROMPTS, null, 2))
                        } catch {
                          alert('Could not load defaults')
                        } finally {
                          setPromptsLoading(false)
                        }
                      }}
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      Reset to Defaults
                    </button>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="btn btn-ghost text-sm"
                        onClick={() => setPromptsEditorOpen(false)}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        className="btn btn-primary text-sm gap-1.5"
                        disabled={promptsSaving}
                        onClick={saveReveriePrompts}
                      >
                        {promptsSaving ? (
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        ) : promptsSaveSuccess ? (
                          <CheckCircle className="w-3.5 h-3.5" />
                        ) : (
                          <Save className="w-3.5 h-3.5" />
                        )}
                        {promptsSaving ? 'Saving…' : promptsSaveSuccess ? 'Saved!' : 'Save'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Agent Info */}
          <div className="card lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <Bot className="w-5 h-5 text-accent" />
              <h3 className="font-display font-semibold text-text-primary">Agent Info</h3>
              <button
                type="button"
                onClick={loadOllamaModels}
                disabled={ollamaModelsLoading}
                className="btn btn-ghost p-1.5 ml-auto"
                title="Refresh Ollama model list"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${ollamaModelsLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 mb-6">
              {[
                { label: 'Agent ID', value: status.agent.id, wide: true },
                { label: 'Name', value: status.agent.name },
                {
                  label: 'Created',
                  value: status.agent.created_at
                    ? new Date(status.agent.created_at).toLocaleString()
                    : 'Unknown',
                },
                { label: 'Tools', value: `${status.agent.tool_count} registered` },
              ].map(item => (
                <div
                  key={item.label}
                  className={`flex items-center justify-between gap-3 py-2 border-b border-surface-200 last:border-0 ${
                    item.wide ? 'sm:col-span-2' : ''
                  }`}
                >
                  <span className="text-sm text-text-secondary shrink-0">{item.label}</span>
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="text-sm text-text-primary font-mono truncate text-right"
                      title={item.value}
                    >
                      {item.value}
                    </span>
                    <button
                      onClick={() => copyToClipboard(item.value, item.label)}
                      className="btn btn-ghost p-1"
                      title="Copy"
                    >
                      {copiedId === item.label ? (
                        <Check className="w-3 h-3 text-success" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="border-t border-surface-200 pt-5 space-y-4">
              <p className="text-sm text-text-secondary">
                Chat and embedding models are served by Ollama. Pick from installed models below.
              </p>

              {ollamaModelsError && (
                <p className="text-xs text-warning">{ollamaModelsError}</p>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-text-primary mb-1.5">
                    Chat model
                  </label>
                  {ollamaModelsLoading && modelHandleOptions.length === 0 ? (
                    <div className="flex items-center gap-2 py-2 text-sm text-text-muted">
                      <div className="w-5 h-5 border-2 border-surface-300 border-t-accent rounded-full animate-spin" />
                      Loading models…
                    </div>
                  ) : (
                    <select
                      className="select"
                      value={selectedModel}
                      onChange={e => {
                        markSettingsDraft()
                        setSelectedModel(e.target.value)
                      }}
                      disabled={modelHandleOptions.length === 0}
                    >
                      {modelHandleOptions.length === 0 ? (
                        <option value="">No models found</option>
                      ) : (
                        modelHandleOptions.map(h => (
                          <option key={h} value={h}>
                            {labelForHandle(h)}
                          </option>
                        ))
                      )}
                    </select>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-text-primary mb-1.5">
                    Embedding model
                  </label>
                  {ollamaModelsLoading && embeddingHandleOptions.length === 0 ? (
                    <div className="flex items-center gap-2 py-2 text-sm text-text-muted">
                      <div className="w-5 h-5 border-2 border-surface-300 border-t-accent rounded-full animate-spin" />
                      Loading models…
                    </div>
                  ) : (
                    <select
                      className="select"
                      value={selectedEmbedding}
                      onChange={e => {
                        markSettingsDraft()
                        setSelectedEmbedding(e.target.value)
                      }}
                      disabled={embeddingHandleOptions.length === 0}
                    >
                      {embeddingHandleOptions.length === 0 ? (
                        <option value="">No models found</option>
                      ) : (
                        embeddingHandleOptions.map(h => (
                          <option key={h} value={h}>
                            {labelForHandle(h)}
                          </option>
                        ))
                      )}
                    </select>
                  )}
                  <p className="text-[11px] text-text-muted mt-1">
                    Use an embedding model (e.g. nomic-embed-text) for archival search.
                  </p>
                </div>
              </div>

              <div className="flex flex-col items-end gap-2 pt-2">
                {updatingModels && modelUpdatePhase && (
                  <p className="text-[11px] text-text-muted text-right max-w-md flex items-center gap-2 justify-end">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin shrink-0" />
                    {modelUpdatePhase}
                  </p>
                )}
                <button
                  type="button"
                  onClick={updateAgentModels}
                  disabled={updatingModels || !selectedModel || !selectedEmbedding}
                  title={
                    modelsConfigDirty
                      ? 'Save model choices to Letta, sync compaction/sleeptime agents, and load in Ollama'
                      : 'Re-apply current models (syncs compaction/sleeptime and reloads Ollama if needed)'
                  }
                  className="btn btn-primary"
                >
                  {updatingModels ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4" />
                  )}
                  {updatingModels
                    ? 'Updating models…'
                    : modelsConfigDirty
                      ? 'Update models'
                      : 'Apply & load in Ollama'}
                </button>
              </div>
            </div>
          </div>

          {/* Tools List */}
          <div className="card lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <Wrench className="w-5 h-5 text-accent" />
              <h3 className="font-display font-semibold text-text-primary">Registered Tools</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {status.agent.tools.map(t => (
                <span key={t} className="text-xs px-2.5 py-1 rounded-lg bg-surface-100 text-text-secondary border border-surface-200">
                  {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
