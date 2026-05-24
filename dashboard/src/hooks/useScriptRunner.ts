import { useState, useCallback } from 'react'
import { authenticatedFetch } from '../utils/api'

export type ScriptStatus = 'idle' | 'running' | 'success' | 'error'

export interface ScriptState {
  status: ScriptStatus
  output: string[]
  startTime?: number
  endTime?: number
}

/**
 * Shared hook for running pipeline scripts and polling their output.
 *
 * Used by both the Pipeline view (App.tsx) and Memory → Maintenance tab.
 * Handles:
 *  - POST to start the script
 *  - Polling /api/mcp/pipeline.output/:id for real-time logs
 *  - Authenticated fetch (OIDC-safe)
 *  - Stop support via /api/mcp/pipeline.stop
 */
export function useScriptRunner(opts?: { onComplete?: (scriptId: string, success: boolean) => void }) {
  const [scriptStates, setScriptStates] = useState<Record<string, ScriptState>>({})

  const runScript = useCallback(
    (scriptId: string, displayName?: string) => {
      const label = displayName || scriptId

      setScriptStates(prev => ({
        ...prev,
        [scriptId]: {
          status: 'running',
          output: [`Starting ${label}...`],
          startTime: Date.now(),
        },
      }))

      // Fire-and-forget: start the script
      authenticatedFetch('/api/mcp/pipeline.run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: scriptId }),
      }).catch(() => {})

      // Poll for real-time output
      const poll = async () => {
        let cursor = 0
        // eslint-disable-next-line no-constant-condition
        while (true) {
          try {
            const res = await authenticatedFetch(
              `/api/mcp/pipeline.output/${scriptId}?cursor=${cursor}`
            )
            const data = await res.json()

            // Script hasn't registered yet — keep waiting
            if (!data.started && !data.done) {
              await new Promise(r => setTimeout(r, 300))
              continue
            }

            for (const { line, index } of data.lines as Array<{ line: string; index: number }>) {
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
              opts?.onComplete?.(scriptId, success)
              return
            }
          } catch {
            /* network hiccup — retry */
          }
          await new Promise(r => setTimeout(r, 250))
        }
      }

      poll()
    },
    [opts?.onComplete]
  )

  const stopScript = useCallback(async (scriptId: string) => {
    try {
      await authenticatedFetch('/api/mcp/pipeline.stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: scriptId }),
      })
    } catch {
      /* ignore */
    }
  }, [])

  /** Merge externally-determined states (e.g. pipeline completion checks). */
  const setStates = setScriptStates

  const hasRunning = Object.values(scriptStates).some(s => s.status === 'running')

  return { scriptStates, setStates, runScript, stopScript, hasRunning }
}
