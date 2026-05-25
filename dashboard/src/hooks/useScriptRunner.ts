import { useState, useCallback, useRef } from 'react'
import { authenticatedFetch } from '../utils/api'
import { appendOutputLines } from '../utils/scriptOutput'

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
 *  - POST to start the script (server returns 202 immediately)
 *  - Polling /api/mcp/pipeline.output/:id for real-time logs
 *  - Authenticated fetch (OIDC-safe)
 *  - Stop support via /api/mcp/pipeline.stop
 */
export function useScriptRunner(opts?: { onComplete?: (scriptId: string, success: boolean) => void }) {
  const [scriptStates, setScriptStates] = useState<Record<string, ScriptState>>({})
  const pollGenerationRef = useRef(0)

  const runScript = useCallback(
    (scriptId: string, displayName?: string) => {
      const label = displayName || scriptId
      const generation = ++pollGenerationRef.current

      setScriptStates(prev => ({
        ...prev,
        [scriptId]: {
          status: 'running',
          output: [`Starting ${label}...`],
          startTime: Date.now(),
        },
      }))

      authenticatedFetch('/api/mcp/pipeline.run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: scriptId }),
      }).catch(() => {})

      const poll = async () => {
        let cursor = 0
        let sawStarted = false

        while (pollGenerationRef.current === generation) {
          try {
            const res = await authenticatedFetch(
              `/api/mcp/pipeline.output/${scriptId}?cursor=${cursor}`
            )
            const data = await res.json()

            if (data.started) {
              sawStarted = true
            }

            if (!data.started && !data.done) {
              await new Promise(r => setTimeout(r, 300))
              continue
            }

            const incoming = (data.lines as Array<{ line: string; index: number }>) || []
            if (incoming.length > 0) {
              const lineTexts = incoming.map(l => l.line)
              const lastIndex = incoming[incoming.length - 1].index
              setScriptStates(prev => ({
                ...prev,
                [scriptId]: {
                  ...prev[scriptId],
                  output: appendOutputLines(prev[scriptId]?.output || [], lineTexts),
                },
              }))
              cursor = lastIndex + 1
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

            if (sawStarted && !data.started && !data.done) {
              const statusRes = await authenticatedFetch('/api/mcp/pipeline.status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: scriptId }),
              })
              if (statusRes.ok) {
                const statusData = await statusRes.json()
                if (!statusData.running) {
                  setScriptStates(prev => ({
                    ...prev,
                    [scriptId]: {
                      ...prev[scriptId],
                      status: 'error',
                      endTime: Date.now(),
                    },
                  }))
                  opts?.onComplete?.(scriptId, false)
                  return
                }
              }
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
    pollGenerationRef.current += 1
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

  const setStates = setScriptStates

  const hasRunning = Object.values(scriptStates).some(s => s.status === 'running')

  return { scriptStates, setStates, runScript, stopScript, hasRunning }
}
