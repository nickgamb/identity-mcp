import { useState, useEffect, useRef } from 'react'
import {
  Play,
  FileText,
  FolderOpen,
  CheckCircle,
  XCircle,
  Clock,
  Terminal,
  Cpu,
  Database,
  Shield,
  Sparkles,
  RefreshCw,
  ChevronRight,
  Eye,
  X,
  Brain,
  Scan
} from 'lucide-react'
import { DataExplorer } from './DataExplorer'
import { MemoryExplorer } from './MemoryExplorer'
import { LibreChatEmbed } from './components/LibreChatEmbed'
import { Sidebar, type MainView } from './components/Sidebar'
import { useAuth } from './auth/AuthContext'
import { LogIn, LogOut, User as UserIcon } from 'lucide-react'
import { authenticatedFetch } from './utils/api'
import { useScriptRunner, type ScriptState } from './hooks/useScriptRunner'
import { EegEnrollmentModal } from './components/eeg/EegEnrollmentModal'
import { EegAuthorizationModal } from './components/eeg/EegAuthorizationModal'
import { BRAND, BrandWordmark } from './brand'

// Script definitions
const SCRIPTS = [
  {
    id: 'parse_conversations',
    name: 'Parse Conversations',
    file: 'parse_conversations.py',
    path: 'scripts/conversation_processing/',
    description:
      'Parses ChatGPT and Anthropic conversations.json and memories.json into JSONL + Markdown per thread.',
    outputs: ['conversations/*.jsonl', 'conversations/*.md'],
    icon: FileText,
    color: 'accent',
    order: 1,
  },
  {
    id: 'parse_memories',
    name: 'Parse Memories',
    file: 'parse_all_memories.py',
    path: 'scripts/conversation_processing/',
    description:
      'Parses ChatGPT memories.json and Claude anthropic_memories.json into searchable JSONL.',
    outputs: ['memory/user.context.jsonl', 'memory/claude.context.jsonl'],
    icon: Database,
    color: 'accent',
    order: 2,
  },
  {
    id: 'analyze_patterns',
    name: 'Analyze Patterns',
    file: 'analyze_patterns.py',
    path: 'scripts/conversation_processing/',
    description:
      'Discovers distinctive terms, topics, entities, and tone patterns from conversations, files, and parsed memories.',
    outputs: ['memory/identity.jsonl', 'memory/patterns.jsonl'],
    icon: Sparkles,
    color: 'accent',
    order: 3,
  },
  {
    id: 'analyze_identity',
    name: 'Analyze Identity',
    file: 'analyze_identity.py',
    path: 'scripts/conversation_processing/',
    description: 'Extracts relational patterns, naming events, and identity momentum from conversations.',
    outputs: ['memory/identity_analysis.jsonl', 'memory/identity_report.md'],
    icon: Cpu,
    color: 'accent',
    order: 4,
  },
  {
    id: 'build_interaction_map',
    name: 'Build Interaction Map',
    file: 'build_interaction_map.py',
    path: 'scripts/conversation_processing/',
    description: 'Indexes conversations and identifies human communication patterns (problem-solving, tempo changes, topic transitions, tone shifts).',
    outputs: ['memory/interaction_map_index.json', 'memory/interaction_key_events.json'],
    icon: FolderOpen,
    color: 'accent',
    order: 5,
  },
  {
    id: 'train_identity_model',
    name: 'Train Identity Model',
    file: 'train_identity_model.py',
    path: 'scripts/identity_model/',
    description:
      'Trains the semantic embedding model from conversation user messages (~60–90 min on CPU; wait for "TRAINING COMPLETE").',
    outputs: ['models/identity/config.json', 'models/identity/identity_centroid.npy', 'models/identity/stylistic_profile.json', 'models/identity/vocabulary_profile.json'],
    icon: Shield,
    color: 'success',
    order: 6,
  },
  {
    id: 'enroll_brainwaves',
    name: 'Enroll Brainwaves',
    file: 'enroll_brainwaves.py',
    path: 'scripts/eeg_identity/',
    description: 'Guides you through neurofeedback tasks while capturing EEG from an EMOTIV Epoc X to build your brainwave identity model.',
    outputs: [
      'models/eeg_identity/config.json',
      'models/eeg_identity/eeg_centroid.npy',
      'models/eeg_identity/spectral_profile.json',
      'models/eeg_identity/accumulated_raw_features.npy',
      'models/eeg_identity/enrollment_history.json',
    ],
    icon: Brain,
    color: 'accent',
    order: 7,
  },
  {
    id: 'authorize_brainwaves',
    name: 'Authorize Brainwaves',
    file: 'authorize_brainwaves.py',
    path: 'scripts/eeg_identity/',
    description: 'Reads live EEG and compares against your enrolled brainwave model to produce an identity assurance signal.',
    outputs: ['models/eeg_identity/config.json'],
    icon: Scan,
    color: 'success',
    order: 8,
  },
]

// ScriptStatus / ScriptState types imported from useScriptRunner hook
type ScriptStatus = 'idle' | 'running' | 'success' | 'error'

function App() {
  const { user, isLoading: authLoading, isAuthenticated, isOidcEnabled, login, logout } = useAuth()
  const [mainView, setMainView] = useState<MainView>('pipeline')
  const { scriptStates, setStates: setScriptStates, runScript: hookRunScript, hasRunning: hasRunningScripts } = useScriptRunner()
  const [selectedScript, setSelectedScript] = useState<string | null>(null)
  const pipelineTerminalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = pipelineTerminalRef.current
    if (!el || !selectedScript) return
    el.scrollTop = el.scrollHeight
  }, [selectedScript, scriptStates[selectedScript ?? '']?.output?.length, scriptStates[selectedScript ?? '']?.status])
  const [fileViewer, setFileViewer] = useState<{ path: string; content: string } | null>(null)
  const [eegModal, setEegModal] = useState<{ type: 'enrollment' | 'authorization' } | null>(null)
  const [mcpStatus, setMcpStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [lettaStatus, setLettaStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [pipelineLoading, setPipelineLoading] = useState(true)
  const pipelineStatusRequestRef = useRef(0)
  const hasRunningScriptsRef = useRef(false)

  useEffect(() => {
    hasRunningScriptsRef.current = hasRunningScripts
  }, [hasRunningScripts])

  useEffect(() => {
    const initialLoad = async () => {
      await Promise.all([checkMcpStatus(), checkLettaStatus(), checkPipelineCompletion()])
      setPipelineLoading(false)
    }
    initialLoad()
    const interval = setInterval(() => {
      checkMcpStatus()
      checkLettaStatus()
      if (!hasRunningScriptsRef.current) {
        checkPipelineCompletion()
      }
    }, 30000)

    const handleDataCleaned = () => { checkPipelineCompletion() }
    window.addEventListener('data-cleaned', handleDataCleaned)
    return () => {
      clearInterval(interval)
      window.removeEventListener('data-cleaned', handleDataCleaned)
    }
  }, [])

  const checkMcpStatus = async () => {
    try {
      const res = await authenticatedFetch('/api/health')
      setMcpStatus(res.ok ? 'online' : 'offline')
    } catch {
      setMcpStatus('offline')
    }
  }

  const checkLettaStatus = async () => {
    try {
      const res = await authenticatedFetch('/api/mcp/letta.status')
      const data = await res.json()
      setLettaStatus(data.available ? 'online' : 'offline')
    } catch {
      setLettaStatus('offline')
    }
  }

  const refreshAllStatus = () => {
    checkMcpStatus()
    checkLettaStatus()
  }

  const isPlaceholderOutput = (output: string[] | undefined) =>
    output?.length === 1 && output[0] === 'Completed previously'

  /** Only show disk-based "Completed previously" when the user has not run the step this session. */
  const shouldApplyArtifactHint = (prev: ScriptState | undefined) => {
    if (!prev || prev.status === 'idle') return true
    if (prev.status === 'running') return false
    if (prev.startTime && prev.startTime > 0) return false
    if (prev.output?.length && !isPlaceholderOutput(prev.output)) return false
    return isPlaceholderOutput(prev.output)
  }

  const applyArtifactHint = (
    next: Record<string, ScriptState>,
    prev: Record<string, ScriptState>,
    scriptId: string,
    artifactPresent: boolean
  ) => {
    if (!artifactPresent) {
      const cur = next[scriptId] ?? prev[scriptId]
      if (cur && isPlaceholderOutput(cur.output)) {
        delete next[scriptId]
      }
      return
    }
    if (shouldApplyArtifactHint(prev[scriptId])) {
      next[scriptId] = {
        status: 'success',
        output: ['Completed previously'],
        startTime: 0,
        endTime: 0,
      }
    }
  }

  const checkPipelineCompletion = async () => {
    const requestId = ++pipelineStatusRequestRef.current
    try {
      const res = await authenticatedFetch('/api/mcp/data.status')
      const data = await res.json()
      if (requestId !== pipelineStatusRequestRef.current) return

      let hasConversations = false
      if (data.counts?.conversationFiles > 0) {
        try {
          const conversationsRes = await authenticatedFetch('/api/mcp/data.conversations')
          if (conversationsRes.ok) {
            const conversationsData = await conversationsRes.json()
            hasConversations =
              Array.isArray(conversationsData.conversations) &&
              conversationsData.conversations.length > 0
          }
        } catch { /* not complete */ }
      }
      if (requestId !== pipelineStatusRequestRef.current) return

      let memoryFileNames: string[] = []
      try {
        const memoryListRes = await authenticatedFetch('/api/mcp/data.memories_list')
        if (memoryListRes.ok) {
          const memoryListData = await memoryListRes.json()
          memoryFileNames = memoryListData.memories?.map((f: { _file: string }) => f._file) || []
        }
      } catch { /* failed */ }
      if (requestId !== pipelineStatusRequestRef.current) return

      setScriptStates(prev => {
        const next: Record<string, ScriptState> = { ...prev }

        applyArtifactHint(next, prev, 'parse_conversations', hasConversations)
        applyArtifactHint(
          next,
          prev,
          'analyze_patterns',
          memoryFileNames.includes('identity.jsonl') && memoryFileNames.includes('patterns.jsonl')
        )
        applyArtifactHint(next, prev, 'parse_memories', memoryFileNames.includes('user.context.jsonl'))
        applyArtifactHint(
          next,
          prev,
          'analyze_identity',
          memoryFileNames.includes('identity_analysis.jsonl')
        )
        applyArtifactHint(next, prev, 'build_interaction_map', !!data.generatedData?.interactionMap)
        applyArtifactHint(next, prev, 'train_identity_model', !!data.generatedData?.identityModel)
        applyArtifactHint(next, prev, 'enroll_brainwaves', !!data.generatedData?.eegIdentityModel)

        return next
      })
    } catch (error) {
      console.error('Failed to check pipeline completion:', error)
    }
  }

  const runScript = (scriptId: string) => {
    const script = SCRIPTS.find(s => s.id === scriptId)
    if (!script) return

    if (scriptId === 'enroll_brainwaves') { setEegModal({ type: 'enrollment' }); return }
    if (scriptId === 'authorize_brainwaves') { setEegModal({ type: 'authorization' }); return }

    setSelectedScript(scriptId)
    hookRunScript(scriptId, script.file)
  }

  const viewFile = async (filePath: string) => {
    try {
      const res = await authenticatedFetch('/api/mcp/file.get', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath: filePath })
      })
      const data = await res.json()
      setFileViewer({
        path: filePath,
        content: data.file?.content ?? data.content ?? JSON.stringify(data, null, 2),
      })
    } catch (error) {
      setFileViewer({ path: filePath, content: `Error loading file: ${error}` })
    }
  }

  const getStatusIcon = (status: ScriptStatus) => {
    switch (status) {
      case 'running': return <RefreshCw className="w-4 h-4 animate-spin text-accent" />
      case 'success': return <CheckCircle className="w-4 h-4 text-success" />
      case 'error': return <XCircle className="w-4 h-4 text-danger" />
      default: return <Clock className="w-4 h-4 text-text-muted" />
    }
  }

  const getStatusBadge = (status: ScriptStatus) => {
    const styles = {
      idle: 'bg-surface-200 text-text-muted',
      running: 'bg-accent/20 text-accent',
      success: 'bg-success/20 text-success',
      error: 'bg-danger/20 text-danger',
    }
    const labels = { idle: 'Ready', running: 'Running', success: 'Complete', error: 'Failed' }
    return <span className={`status-badge ${styles[status]}`}>{labels[status]}</span>
  }

  const handleViewChange = (view: MainView) => {
    if (hasRunningScripts && mainView === 'pipeline') {
      if (!confirm('Scripts are running. Switching views may lose progress output. Continue?')) return
    }
    setMainView(view)
  }

  const authGate = (
    <div className="flex flex-col items-center justify-center py-24">
      <div className="card max-w-md text-center">
        <img
          src={BRAND.logoSrc}
          alt={BRAND.company}
          className="h-12 w-12 mx-auto mb-4 object-contain opacity-90"
        />
        <h2 className="font-display text-xl font-semibold text-text-primary mb-2">Authentication Required</h2>
        <p className="text-text-secondary mb-6">Please log in to access the dashboard.</p>
        <button onClick={login} className="btn btn-primary">
          <LogIn className="w-4 h-4" /><span>Login</span>
        </button>
      </div>
    </div>
  )

  const requiresAuth = isOidcEnabled && !isAuthenticated

  return (
    <div className="flex flex-col h-screen bg-surface overflow-hidden">
      {/* ── Top bar (full width) ─────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-surface-200 bg-surface-50/80 backdrop-blur-sm shrink-0 z-10">
        {/* Left: brand */}
        <div className="flex items-center gap-3 min-w-0">
          <img
            src={BRAND.logoSrc}
            alt={BRAND.company}
            className="h-8 w-8 shrink-0 object-contain"
          />
          <div className="h-8 w-px bg-surface-200 shrink-0" aria-hidden />
          <div className="min-w-0">
            <h1 className="text-sm text-text-primary leading-tight">
              <BrandWordmark />
            </h1>
            <p className="text-[10px] text-text-muted leading-tight truncate">{BRAND.tagline}</p>
          </div>
        </div>

        {/* Right: auth */}
        <div className="flex items-center gap-3">
          {!authLoading && isOidcEnabled && (
            <>
              {isAuthenticated && user ? (
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-100">
                    <UserIcon className="w-4 h-4 text-text-secondary" />
                    <span className="text-sm text-text-primary">
                      {user.profile?.preferred_username || user.profile?.email || user.profile?.sub || 'User'}
                    </span>
                  </div>
                  <button onClick={logout} className="btn btn-ghost p-1.5" title="Logout">
                    <LogOut className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <button onClick={login} className="btn btn-primary text-sm">
                  <LogIn className="w-4 h-4" /><span>Login</span>
                </button>
              )}
            </>
          )}
        </div>
      </header>

      {/* ── Body: sidebar + content ──────────────────────────────── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Sidebar
          activeView={mainView}
          onViewChange={handleViewChange}
          mcpStatus={mcpStatus}
          lettaStatus={lettaStatus}
          onRefreshStatus={refreshAllStatus}
          hasRunningScripts={hasRunningScripts}
        />

        {/* Content */}
        <main
          className={`flex-1 min-h-0 ${
            mainView === 'chat' ? 'flex flex-col overflow-hidden' : 'overflow-y-auto'
          }`}
        >
          {mainView === 'chat' ? (
            requiresAuth ? (
              <div className="flex flex-1 items-center justify-center p-6">{authGate}</div>
            ) : (
              <LibreChatEmbed />
            )
          ) : (
          <div className="max-w-[1600px] mx-auto px-6 py-6">

            {/* Memory Explorer */}
            {mainView === 'memory' && (requiresAuth ? authGate : <MemoryExplorer />)}

            {/* Data Explorer */}
            {mainView === 'data' && (requiresAuth ? authGate : <DataExplorer />)}

            {/* Pipeline */}
            {mainView === 'pipeline' && (
              requiresAuth ? authGate : pipelineLoading ? (
                <div className="flex flex-col items-center justify-center py-24 gap-4">
                  <div className="w-10 h-10 border-4 border-surface-300 border-t-accent rounded-full animate-spin" />
                  <p className="text-text-muted text-sm">Loading pipeline status...</p>
                </div>
              ) : (
              <>
                <section className="mb-8">
                  <h2 className="font-display text-lg font-semibold text-text-primary mb-4">Processing Pipeline</h2>
                  <p className="text-text-secondary mb-6">
                    Run these scripts in order to process your ChatGPT and Anthropic formatted conversation export and train your identity model.
                  </p>
                  <div className="flex items-center gap-2 mb-8 overflow-x-auto pb-2">
                    {SCRIPTS.sort((a, b) => a.order - b.order).map((script, idx) => (
                      <div key={script.id} className="flex items-center">
                        <button
                          onClick={() => setSelectedScript(script.id)}
                          className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                            selectedScript === script.id
                              ? 'bg-accent/20 text-accent border border-accent/30'
                              : 'bg-surface-100 text-text-secondary hover:bg-surface-200'
                          }`}
                        >
                          <span className="text-xs font-medium">{idx + 1}</span>
                          <span className="text-sm whitespace-nowrap">{script.name}</span>
                          {getStatusIcon(scriptStates[script.id]?.status || 'idle')}
                        </button>
                        {idx < SCRIPTS.length - 1 && (
                          <ChevronRight className="w-4 h-4 text-surface-300 mx-1 flex-shrink-0" />
                        )}
                      </div>
                    ))}
                  </div>
                </section>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2 space-y-4">
                    {SCRIPTS.sort((a, b) => a.order - b.order).map((script) => {
                      const Icon = script.icon
                      const state = scriptStates[script.id] || { status: 'idle' as ScriptStatus, output: [] }
                      const isSelected = selectedScript === script.id
                      return (
                        <div
                          key={script.id}
                          className={`card cursor-pointer ${isSelected ? 'border-accent shadow-glow-accent' : ''}`}
                          onClick={() => setSelectedScript(script.id)}
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-3">
                              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                                script.color === 'success' ? 'bg-success/20' : 'bg-accent/20'
                              }`}>
                                <Icon className={`w-5 h-5 ${script.color === 'success' ? 'text-success' : 'text-accent'}`} />
                              </div>
                              <div>
                                <h3 className="font-display font-semibold text-text-primary">{script.name}</h3>
                                <p className="text-xs text-text-muted font-mono">{script.path}{script.file}</p>
                              </div>
                            </div>
                            {getStatusBadge(state.status)}
                          </div>
                          <p className="text-sm text-text-secondary mb-4">{script.description}</p>
                          <div className="flex items-center justify-between">
                            <div className="flex flex-wrap gap-2">
                              {script.outputs.map((output) => (
                                <button
                                  key={output}
                                  onClick={(e) => { e.stopPropagation(); viewFile(output) }}
                                  className="text-xs px-2 py-1 rounded bg-surface-100 text-text-muted hover:bg-surface-200 hover:text-text-primary transition-colors flex items-center gap-1"
                                >
                                  <Eye className="w-3 h-3" />
                                  {output.split('/').pop()}
                                </button>
                              ))}
                            </div>
                            <button
                              onClick={(e) => { e.stopPropagation(); runScript(script.id) }}
                              disabled={state.status === 'running'}
                              className={`btn ${state.status === 'running' ? 'btn-ghost cursor-not-allowed' : 'btn-primary'}`}
                            >
                              {state.status === 'running' ? (
                                <><RefreshCw className="w-4 h-4 animate-spin" />Running</>
                              ) : (
                                <><Play className="w-4 h-4" />Run</>
                              )}
                            </button>
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  <div className="lg:col-span-1">
                    <div className="card sticky top-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Terminal className="w-5 h-5 text-accent" />
                        <h3 className="font-display font-semibold text-text-primary">Output</h3>
                      </div>
                      {selectedScript ? (
                        <>
                          <div ref={pipelineTerminalRef} className="terminal max-h-[500px] overflow-y-auto">
                            {scriptStates[selectedScript]?.output && scriptStates[selectedScript].output.length > 0 ? (
                              scriptStates[selectedScript].output.map((line, idx) => (
                                <div key={idx} className="terminal-line stdout">{line}</div>
                              ))
                            ) : (
                              <div className="terminal-line text-text-muted italic">No output yet. Run the script to see output.</div>
                            )}
                            {scriptStates[selectedScript]?.status === 'running' && (
                              <div className="terminal-line text-accent animate-pulse">&#x25CB;</div>
                            )}
                          </div>
                          {scriptStates[selectedScript]?.status === 'running' && (
                            <div className="mt-2 text-xs text-text-muted flex items-center gap-2">
                              <RefreshCw className="w-3 h-3 animate-spin" />
                              Running... {scriptStates[selectedScript]?.startTime && (
                                <span>({Math.floor((Date.now() - scriptStates[selectedScript].startTime!) / 1000)}s elapsed)</span>
                              )}
                            </div>
                          )}
                          {scriptStates[selectedScript]?.endTime &&
                            (scriptStates[selectedScript].startTime ?? 0) > 0 && (
                            <div className="mt-2 text-xs text-text-muted">
                              {scriptStates[selectedScript].status === 'success' ? (
                                <span className="text-success">Completed</span>
                              ) : scriptStates[selectedScript].status === 'error' ? (
                                <span className="text-danger">Failed</span>
                              ) : null}
                              {' '}in {((scriptStates[selectedScript].endTime! - scriptStates[selectedScript].startTime!) / 1000).toFixed(1)}s
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="text-center py-12 text-text-muted">
                          <Terminal className="w-12 h-12 mx-auto mb-3 opacity-30" />
                          <p>Select a script to see output</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
              )
            )}
          </div>
          )}
        </main>
      </div>

      {/* File Viewer Modal */}
      {fileViewer && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-surface-50 rounded-xl border border-surface-200 w-full max-w-4xl max-h-[80vh] flex flex-col animate-fade-in">
            <div className="flex items-center justify-between p-4 border-b border-surface-200">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-accent" />
                <span className="font-mono text-sm text-text-primary">{fileViewer.path}</span>
              </div>
              <button onClick={() => setFileViewer(null)} className="btn btn-ghost"><X className="w-5 h-5" /></button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <pre className="font-mono text-sm text-text-secondary whitespace-pre-wrap">{fileViewer.content}</pre>
            </div>
          </div>
        </div>
      )}

      {eegModal?.type === 'enrollment' && (
        <EegEnrollmentModal onClose={() => { setEegModal(null); checkPipelineCompletion() }} authenticatedFetch={authenticatedFetch} />
      )}
      {eegModal?.type === 'authorization' && (
        <EegAuthorizationModal onClose={() => { setEegModal(null) }} authenticatedFetch={authenticatedFetch} />
      )}
    </div>
  )
}

export default App
