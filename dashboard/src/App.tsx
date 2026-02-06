import { useState, useEffect } from 'react'
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
  LayoutDashboard,
  Brain,
  Scan
} from 'lucide-react'
import { DataExplorer } from './DataExplorer'
import { useAuth } from './auth/AuthContext'
import { LogIn, LogOut, User as UserIcon } from 'lucide-react'
import { authenticatedFetch } from './utils/api'
import { EegEnrollmentModal } from './components/eeg/EegEnrollmentModal'
import { EegAuthorizationModal } from './components/eeg/EegAuthorizationModal'

// Script definitions
const SCRIPTS = [
  {
    id: 'parse_conversations',
    name: 'Parse Conversations',
    file: 'parse_conversations.py',
    path: 'scripts/conversation_processing/',
    description: 'Converts raw conversations.json into structured JSONL files for each conversation.',
    outputs: ['conversations/*.jsonl', 'conversations/*.md'],
    icon: FileText,
    color: 'accent',
    order: 1,
  },
  {
    id: 'analyze_patterns',
    name: 'Analyze Patterns',
    file: 'analyze_patterns.py',
    path: 'scripts/conversation_processing/',
    description: 'Discovers distinctive terms, topics, entities, and tone patterns from your conversations.',
    outputs: ['memory/identity.jsonl', 'memory/patterns.jsonl'],
    icon: Sparkles,
    color: 'accent',
    order: 2,
  },
  {
    id: 'parse_memories',
    name: 'Parse Memories',
    file: 'parse_memories.py',
    path: 'scripts/conversation_processing/',
    description: 'Converts ChatGPT memories.json into searchable context records.',
    outputs: ['memory/user.context.jsonl'],
    icon: Database,
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
    description: 'Trains the semantic embedding model using all processed data. Creates your identity fingerprint.',
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
    outputs: ['models/eeg_identity/config.json', 'models/eeg_identity/eeg_centroid.npy', 'models/eeg_identity/spectral_profile.json'],
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

type ScriptStatus = 'idle' | 'running' | 'success' | 'error'

interface ScriptState {
  status: ScriptStatus
  output: string[]
  startTime?: number
  endTime?: number
}

type MainView = 'pipeline' | 'data'

function App() {
  const { user, isLoading: authLoading, isAuthenticated, isOidcEnabled, login, logout } = useAuth()
  const [mainView, setMainView] = useState<MainView>('pipeline')
  const [scriptStates, setScriptStates] = useState<Record<string, ScriptState>>({})
  const [selectedScript, setSelectedScript] = useState<string | null>(null)
  const [fileViewer, setFileViewer] = useState<{ path: string; content: string } | null>(null)
  const [eegModal, setEegModal] = useState<{ type: 'enrollment' | 'authorization' } | null>(null)
  const [mcpStatus, setMcpStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [pipelineLoading, setPipelineLoading] = useState(true)
  
  // Check if any scripts are running
  const hasRunningScripts = Object.values(scriptStates).some(state => state.status === 'running')

  // Check MCP status and pipeline completion on mount
  useEffect(() => {
    const initialLoad = async () => {
      await Promise.all([checkMcpStatus(), checkPipelineCompletion()])
      setPipelineLoading(false)
    }
    initialLoad()
    const interval = setInterval(() => {
      checkMcpStatus()
      checkPipelineCompletion()
    }, 30000)
    
    // Listen for data-cleaned event from DataExplorer
    const handleDataCleaned = () => {
      checkPipelineCompletion()
    }
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

  const checkPipelineCompletion = async () => {
    try {
      // Get current state FIRST to preserve running scripts
      let runningScripts: Record<string, ScriptState> = {}
      setScriptStates(prev => {
        runningScripts = {}
        for (const [scriptId, state] of Object.entries(prev)) {
          if (state.status === 'running') {
            runningScripts[scriptId] = state
          }
        }
        return prev // Don't change state yet
      })
      
      // Check data status to determine which scripts have completed
      const res = await authenticatedFetch('/api/mcp/data.status')
      const data = await res.json()
      
      // Start with running scripts - only add completed if files exist
      const finalStates: Record<string, ScriptState> = { ...runningScripts }
      
      // Parse Conversations - check if conversation JSONL files exist
      if (data.counts?.conversationFiles > 0) {
        try {
          const conversationsRes = await authenticatedFetch('/api/mcp/data.conversations')
          if (conversationsRes.ok) {
            const conversationsData = await conversationsRes.json()
            if (conversationsData.conversations && conversationsData.conversations.length > 0) {
              finalStates['parse_conversations'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
            }
          }
        } catch (e) {
          // File check failed, not complete - don't add to finalStates
        }
      }
      // If no files or check failed, script won't be in finalStates = shows as Ready
      
      // Check memory files using the same API the dashboard uses
      try {
        const memoryListRes = await authenticatedFetch('/api/mcp/data.memories_list')
        if (memoryListRes.ok) {
          const memoryListData = await memoryListRes.json()
          const memoryFileNames = memoryListData.memories?.map((f: any) => f._file) || []
          
          // Analyze Patterns - check if identity.jsonl and patterns.jsonl exist
          if (memoryFileNames.includes('identity.jsonl') && memoryFileNames.includes('patterns.jsonl')) {
            finalStates['analyze_patterns'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
          }
          
          // Parse Memories - check if user.context.jsonl exists
          if (memoryFileNames.includes('user.context.jsonl')) {
            finalStates['parse_memories'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
          }
          
          // Analyze Identity - check if identity_analysis.jsonl exists
          if (memoryFileNames.includes('identity_analysis.jsonl')) {
            finalStates['analyze_identity'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
          }
        }
      } catch (e) {
        // Memory list check failed
      }
      
      // Build Interaction Map - check via data.status
      if (data.generatedData?.interactionMap) {
        finalStates['build_interaction_map'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
      }
      
      // Train Identity Model - use data.status which already checks this
      if (data.generatedData?.identityModel) {
        finalStates['train_identity_model'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
      }
      
      // Enroll Brainwaves - check if EEG identity model exists
      if (data.generatedData?.eegIdentityModel) {
        finalStates['enroll_brainwaves'] = { status: 'success', output: ['Completed previously'], startTime: 0, endTime: 0 }
      }
      
      // COMPLETELY REPLACE state - only scripts with running status or completed files will be in finalStates
      // Scripts without files won't be in finalStates, so they'll show as Ready (idle)
      setScriptStates(finalStates)
    } catch (error) {
      console.error('Failed to check pipeline completion:', error)
    }
  }

  const runScript = async (scriptId: string) => {
    const script = SCRIPTS.find(s => s.id === scriptId)
    if (!script) return

    // ── EEG scripts get a dedicated visual modal ──
    if (scriptId === 'enroll_brainwaves') {
      setEegModal({ type: 'enrollment' })
      return
    }
    if (scriptId === 'authorize_brainwaves') {
      setEegModal({ type: 'authorization' })
      return
    }

    // Clear output and mark running
    setScriptStates(prev => ({
      ...prev,
      [scriptId]: { status: 'running', output: [`Starting ${script.file}...`], startTime: Date.now() }
    }))
    setSelectedScript(scriptId)

    // Fire-and-forget: start the script (response arrives when it finishes)
    authenticatedFetch('/api/mcp/pipeline.run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ script: script.id })
    }).catch(() => { /* SSE handles output and completion */ })

    // Poll for real-time output (SSE doesn't work through Vite's dev proxy)
    const pollOutput = async () => {
      let cursor = 0

      // eslint-disable-next-line no-constant-condition
      while (true) {
        try {
          const res = await fetch(`/api/mcp/pipeline.output/${scriptId}?cursor=${cursor}`)
          const data = await res.json()

          // Script hasn't registered yet — keep waiting
          if (!data.started && !data.done) {
            await new Promise(r => setTimeout(r, 300))
            continue
          }

          for (const { line, index } of data.lines) {
            setScriptStates(prev => ({
              ...prev,
              [scriptId]: {
                ...prev[scriptId],
                output: [...(prev[scriptId]?.output || []), line],
              }
            }))
            cursor = index + 1
          }

          if (data.done) {
            const success = data.exitCode === 0
            setScriptStates(prev => ({
              ...prev,
              [scriptId]: {
                ...prev[scriptId],
                status: success ? 'success' : 'error',
                endTime: Date.now(),
              }
            }))
            return
          }
        } catch {
          // Network hiccup — retry
        }
        await new Promise(r => setTimeout(r, 250))
      }
    }
    pollOutput()
  }

  const viewFile = async (filePath: string) => {
    try {
      const res = await authenticatedFetch('/api/mcp/file.get', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath: filePath })
      })
      const data = await res.json()
      setFileViewer({ path: filePath, content: data.content || JSON.stringify(data, null, 2) })
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
    const labels = {
      idle: 'Ready',
      running: 'Running',
      success: 'Complete',
      error: 'Failed',
    }
    return (
      <span className={`status-badge ${styles[status]}`}>
        {labels[status]}
      </span>
    )
  }

  return (
    <div className="min-h-screen bg-surface">
      {/* Header */}
      <header className="border-b border-surface-200 bg-surface-50/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-accent-bright flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-display text-xl font-semibold text-text-primary">Identity MCP</h1>
              <p className="text-xs text-text-muted">Processing Dashboard</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                mcpStatus === 'online' ? 'bg-success' : 
                mcpStatus === 'offline' ? 'bg-danger' : 'bg-warning animate-pulse'
              }`} />
              <span className="text-sm text-text-secondary">
                MCP {mcpStatus === 'online' ? 'Online' : mcpStatus === 'offline' ? 'Offline' : 'Checking...'}
              </span>
            </div>
            <button onClick={checkMcpStatus} className="btn btn-ghost">
              <RefreshCw className="w-4 h-4" />
            </button>
            
            {/* Auth UI */}
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
                    <button onClick={logout} className="btn btn-ghost" title="Logout">
                      <LogOut className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <button onClick={login} className="btn btn-primary" title="Login">
                    <LogIn className="w-4 h-4" />
                    <span>Login</span>
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-8 py-8">
        {/* View Switcher */}
        <div className="flex items-center gap-2 mb-6">
          <button
            onClick={() => {
              if (hasRunningScripts && mainView === 'pipeline') {
                if (!confirm('Scripts are running. Switching views may lose progress output. Continue?')) {
                  return
                }
              }
              setMainView('pipeline')
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              mainView === 'pipeline'
                ? 'bg-accent text-white'
                : 'bg-surface-100 text-text-secondary hover:bg-surface-200'
            }`}
          >
            <Terminal className="w-4 h-4" />
            Pipeline
            {hasRunningScripts && mainView === 'pipeline' && (
              <RefreshCw className="w-3 h-3 animate-spin" />
            )}
          </button>
          <button
            onClick={() => {
              if (hasRunningScripts && mainView === 'pipeline') {
                if (!confirm('Scripts are running. Switching views may lose progress output. Continue?')) {
                  return
                }
              }
              setMainView('data')
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              mainView === 'data'
                ? 'bg-accent text-white'
                : 'bg-surface-100 text-text-secondary hover:bg-surface-200'
            }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            Data Explorer
          </button>
        </div>

        {/* Data Explorer View */}
        {mainView === 'data' && (
          isOidcEnabled && !isAuthenticated ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="card max-w-md text-center">
                <Shield className="w-16 h-16 mx-auto mb-4 text-accent opacity-50" />
                <h2 className="font-display text-xl font-semibold text-text-primary mb-2">Authentication Required</h2>
                <p className="text-text-secondary mb-6">
                  Please log in to access the Data Explorer.
                </p>
                <button onClick={login} className="btn btn-primary">
                  <LogIn className="w-4 h-4" />
                  <span>Login</span>
                </button>
              </div>
            </div>
          ) : (
            <DataExplorer />
          )
        )}

        {/* Pipeline View */}
        {mainView === 'pipeline' && (
          isOidcEnabled && !isAuthenticated ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="card max-w-md text-center">
                <Shield className="w-16 h-16 mx-auto mb-4 text-accent opacity-50" />
                <h2 className="font-display text-xl font-semibold text-text-primary mb-2">Authentication Required</h2>
                <p className="text-text-secondary mb-6">
                  Please log in to access the Processing Pipeline.
                </p>
                <button onClick={login} className="btn btn-primary">
                  <LogIn className="w-4 h-4" />
                  <span>Login</span>
                </button>
              </div>
            </div>
          ) : pipelineLoading ? (
            <div className="flex flex-col items-center justify-center py-24 gap-4">
              <div className="w-10 h-10 border-4 border-surface-300 border-t-accent rounded-full animate-spin" />
              <p className="text-text-muted text-sm">Loading pipeline status...</p>
            </div>
          ) : (
          <>
            {/* Pipeline Overview */}
            <section className="mb-8">
          <h2 className="font-display text-lg font-semibold text-text-primary mb-4">Processing Pipeline</h2>
          <p className="text-text-secondary mb-6">
            Run these scripts in order to process your ChatGPT formatted conversation export and train your identity model.
          </p>
          
          {/* Pipeline Steps */}
          <div className="flex items-center gap-2 mb-8 overflow-x-auto pb-2">
            {SCRIPTS.sort((a, b) => a.order - b.order).map((script, idx) => (
              <div key={script.id} className="flex items-center">
                <button
                  onClick={() => {
                    // Just change selection - don't clear output
                    setSelectedScript(script.id)
                  }}
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
          {/* Script Cards */}
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
                        <Icon className={`w-5 h-5 ${
                          script.color === 'success' ? 'text-success' : 'text-accent'
                        }`} />
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
                          onClick={(e) => { e.stopPropagation(); viewFile(output); }}
                          className="text-xs px-2 py-1 rounded bg-surface-100 text-text-muted hover:bg-surface-200 hover:text-text-primary transition-colors flex items-center gap-1"
                        >
                          <Eye className="w-3 h-3" />
                          {output.split('/').pop()}
                        </button>
                      ))}
                    </div>
                    
                    <button
                      onClick={(e) => { e.stopPropagation(); runScript(script.id); }}
                      disabled={state.status === 'running'}
                      className={`btn ${state.status === 'running' ? 'btn-ghost cursor-not-allowed' : 'btn-primary'}`}
                    >
                      {state.status === 'running' ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Running
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Run
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Output Panel */}
          <div className="lg:col-span-1">
            <div className="card sticky top-24">
              <div className="flex items-center gap-2 mb-4">
                <Terminal className="w-5 h-5 text-accent" />
                <h3 className="font-display font-semibold text-text-primary">Output</h3>
              </div>
              
              {selectedScript ? (
                <>
                  <div className="terminal max-h-[500px] overflow-y-auto">
                    {scriptStates[selectedScript]?.output && scriptStates[selectedScript].output.length > 0 ? (
                      scriptStates[selectedScript].output.map((line, idx) => (
                        <div key={idx} className="terminal-line stdout">{line}</div>
                      ))
                    ) : (
                      <div className="terminal-line text-text-muted italic">No output yet. Run the script to see output.</div>
                    )}
                    {scriptStates[selectedScript]?.status === 'running' && (
                      <div className="terminal-line text-accent animate-pulse">▋</div>
                    )}
                  </div>
                  {selectedScript && scriptStates[selectedScript]?.status === 'running' && (
                    <div className="mt-2 text-xs text-text-muted flex items-center gap-2">
                      <RefreshCw className="w-3 h-3 animate-spin" />
                      Running... {scriptStates[selectedScript]?.startTime && (
                        <span>
                          ({Math.floor((Date.now() - scriptStates[selectedScript].startTime!) / 1000)}s elapsed)
                        </span>
                      )}
                    </div>
                  )}
                  {selectedScript && scriptStates[selectedScript]?.endTime && (
                    <div className="mt-2 text-xs text-text-muted">
                      {scriptStates[selectedScript].status === 'success' ? (
                        <span className="text-success">✓ Completed</span>
                      ) : scriptStates[selectedScript].status === 'error' ? (
                        <span className="text-danger">✗ Failed</span>
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
      </main>

      {/* File Viewer Modal */}
      {fileViewer && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-surface-50 rounded-xl border border-surface-200 w-full max-w-4xl max-h-[80vh] flex flex-col animate-fade-in">
            <div className="flex items-center justify-between p-4 border-b border-surface-200">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-accent" />
                <span className="font-mono text-sm text-text-primary">{fileViewer.path}</span>
              </div>
              <button onClick={() => setFileViewer(null)} className="btn btn-ghost">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              <pre className="font-mono text-sm text-text-secondary whitespace-pre-wrap">
                {fileViewer.content}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* EEG Enrollment Modal */}
      {eegModal?.type === 'enrollment' && (
        <EegEnrollmentModal
          onClose={() => {
            setEegModal(null)
            checkPipelineCompletion()
          }}
          authenticatedFetch={authenticatedFetch}
        />
      )}

      {/* EEG Authorization Modal */}
      {eegModal?.type === 'authorization' && (
        <EegAuthorizationModal
          onClose={() => {
            setEegModal(null)
          }}
          authenticatedFetch={authenticatedFetch}
        />
      )}
    </div>
  )
}

export default App

