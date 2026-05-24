import { useState } from 'react'
import {
  Play,
  RefreshCw,
  Terminal,
  CheckCircle,
  XCircle,
  Clock,
  Database,
  Wrench,
  Upload,
  Sparkles,
  Square,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { authenticatedFetch } from '../utils/api'

type ScriptStatus = 'idle' | 'running' | 'success' | 'error'

interface ScriptState {
  status: ScriptStatus
  output: string[]
  startTime?: number
  endTime?: number
}

interface MaintenanceScript {
  id: string
  name: string
  file: string
  path: string
  description: string
  icon: LucideIcon
  durationHint?: string
}

const MAINTENANCE_SCRIPTS: MaintenanceScript[] = [
  {
    id: 'letta_register_tools',
    name: 'Register MCP Tools',
    file: 'register_tools.py',
    path: 'letta/',
    description:
      'Attach identity-mcp tools on the Letta agent so chat and sleeptime can search files, conversations, and the full corpus.',
    icon: Wrench,
    durationHint: '~1 min',
  },
  {
    id: 'letta_ingest',
    name: 'Ingest Corpus → Archival',
    file: 'ingest.py',
    path: 'letta/',
    description:
      'Embed conversations, memory JSONL, and files/ into Letta archival memory. Skips content already ingested (hash dedupe). Can take hours on large corpora.',
    icon: Database,
    durationHint: 'long',
  },
  {
    id: 'letta_ingest_init',
    name: 'Agent Self-Init',
    file: 'ingest.py --init-only',
    path: 'letta/',
    description:
      'Prompts the agent to explore archival memory and rewrite persona + human blocks from what it finds. Run after a large ingest.',
    icon: Sparkles,
    durationHint: '~5 min',
  },
  {
    id: 'letta_bootstrap_persona',
    name: 'Seed Persona from identity.jsonl',
    file: 'bootstrap_agent.py --skip-archival',
    path: 'letta/',
    description:
      'Ensures the agent exists and updates the persona block from memory/identity.jsonl without re-inserting archival passages.',
    icon: Upload,
    durationHint: '~1 min',
  },
]

function statusIcon(status: ScriptStatus) {
  switch (status) {
    case 'running':
      return <RefreshCw className="w-4 h-4 animate-spin text-accent" />
    case 'success':
      return <CheckCircle className="w-4 h-4 text-success" />
    case 'error':
      return <XCircle className="w-4 h-4 text-danger" />
    default:
      return <Clock className="w-4 h-4 text-text-muted" />
  }
}

function statusBadge(status: ScriptStatus) {
  const styles = {
    idle: 'bg-surface-200 text-text-muted',
    running: 'bg-accent/20 text-accent',
    success: 'bg-success/20 text-success',
    error: 'bg-danger/20 text-danger',
  }
  const labels = { idle: 'Ready', running: 'Running', success: 'Complete', error: 'Failed' }
  return <span className={`status-badge ${styles[status]}`}>{labels[status]}</span>
}

interface MemoryMaintenanceProps {
  onJobComplete?: () => void
}

export function MemoryMaintenance({ onJobComplete }: MemoryMaintenanceProps) {
  const [scriptStates, setScriptStates] = useState<Record<string, ScriptState>>({})
  const [selectedScript, setSelectedScript] = useState<string>(
    MAINTENANCE_SCRIPTS[0].id
  )

  const runScript = async (scriptId: string) => {
    const script = MAINTENANCE_SCRIPTS.find(s => s.id === scriptId)
    if (!script) return

    setScriptStates(prev => ({
      ...prev,
      [scriptId]: {
        status: 'running',
        output: [`Starting ${script.path}${script.file}...`],
        startTime: Date.now(),
      },
    }))
    setSelectedScript(scriptId)

    authenticatedFetch('/api/mcp/pipeline.run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ script: scriptId }),
    }).catch(() => {})

    let cursor = 0
    // eslint-disable-next-line no-constant-condition
    while (true) {
      try {
        const res = await fetch(`/api/mcp/pipeline.output/${scriptId}?cursor=${cursor}`)
        const data = await res.json()
        if (!data.started && !data.done) {
          await new Promise(r => setTimeout(r, 300))
          continue
        }
        for (const { line, index }: { line: string; index: number } of data.lines) {
          setScriptStates(prev => ({
            ...prev,
            [scriptId]: {
              ...prev[scriptId],
              output: [...(prev[scriptId]?.output || []), line],
            },
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
            },
          }))
          if (success) onJobComplete?.()
          return
        }
      } catch {
        /* retry */
      }
      await new Promise(r => setTimeout(r, 250))
    }
  }

  const stopScript = async (scriptId: string) => {
    try {
      await authenticatedFetch('/api/mcp/pipeline.stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: scriptId }),
      })
    } catch {
      /* ignore */
    }
  }

  const selectedState = selectedScript ? scriptStates[selectedScript] : undefined

  return (
    <div>
      <p className="text-text-secondary text-sm mb-6 max-w-3xl">
        Letta maintenance jobs — register tools, ingest into archival memory, self-init, or
        re-seed persona. For parsing conversations and training the identity model, use the
        main Pipeline view. Re-run ingest after large uploads; re-run register tools after tool
        changes.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {MAINTENANCE_SCRIPTS.map(script => {
            const Icon = script.icon
            const state = scriptStates[script.id] || {
              status: 'idle' as ScriptStatus,
              output: [],
            }
            const isSelected = selectedScript === script.id
            return (
              <div
                key={script.id}
                className={`card cursor-pointer ${isSelected ? 'border-accent shadow-glow-accent' : ''}`}
                onClick={() => setSelectedScript(script.id)}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-accent/20 flex items-center justify-center">
                      <Icon className="w-5 h-5 text-accent" />
                    </div>
                    <div>
                      <h4 className="font-display font-semibold text-text-primary">{script.name}</h4>
                      <p className="text-xs text-text-muted font-mono">
                        {script.path}
                        {script.file}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {statusIcon(state.status)}
                    {statusBadge(state.status)}
                  </div>
                </div>
                <p className="text-sm text-text-secondary mb-3">{script.description}</p>
                {script.durationHint && (
                  <p className="text-[11px] text-text-muted mb-3">
                    Typical duration: {script.durationHint}
                  </p>
                )}
                <div className="flex items-center justify-end gap-2">
                  {state.status === 'running' && (
                    <button
                      type="button"
                      onClick={e => {
                        e.stopPropagation()
                        stopScript(script.id)
                      }}
                      className="btn btn-ghost text-sm"
                    >
                      <Square className="w-3.5 h-3.5" />
                      Stop
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={e => {
                      e.stopPropagation()
                      runScript(script.id)
                    }}
                    disabled={state.status === 'running'}
                    className={`btn text-sm ${
                      state.status === 'running' ? 'btn-ghost cursor-not-allowed' : 'btn-primary'
                    }`}
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

        <div className="lg:col-span-1">
          <div className="card sticky top-6">
            <div className="flex items-center gap-2 mb-4">
              <Terminal className="w-5 h-5 text-accent" />
              <h3 className="font-display font-semibold text-text-primary">Output</h3>
            </div>
            {selectedScript ? (
              <>
                <p className="text-xs text-text-muted mb-2 font-mono">{selectedScript}</p>
                <div className="terminal max-h-[500px] overflow-y-auto">
                  {selectedState?.output && selectedState.output.length > 0 ? (
                    selectedState.output.map((line, idx) => (
                      <div key={idx} className="terminal-line stdout">
                        {line}
                      </div>
                    ))
                  ) : (
                    <div className="terminal-line text-text-muted italic">
                      No output yet. Run a script to see logs.
                    </div>
                  )}
                  {selectedState?.status === 'running' && (
                    <div className="terminal-line text-accent animate-pulse">&#x25CB;</div>
                  )}
                </div>
                {selectedState?.status === 'running' && selectedState.startTime && (
                  <div className="mt-2 text-xs text-text-muted flex items-center gap-2">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    Running… ({Math.floor((Date.now() - selectedState.startTime) / 1000)}s)
                  </div>
                )}
                {selectedState?.endTime && (
                  <div className="mt-2 text-xs text-text-muted">
                    {selectedState.status === 'success' ? (
                      <span className="text-success">Completed</span>
                    ) : (
                      <span className="text-danger">Failed</span>
                    )}{' '}
                    in{' '}
                    {(
                      (selectedState.endTime - (selectedState.startTime || selectedState.endTime)) /
                      1000
                    ).toFixed(1)}
                    s
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12 text-text-muted">
                <Terminal className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>Select a script</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
