/**
 * EEG Enrollment Modal
 *
 * Full-screen guided enrollment experience with visual stimuli:
 *  - Checkerboard   (baseline/resting)
 *  - Breathing circle (meditation)
 *  - Word display    (cognitive word-focus)
 *  - Instruction text (motor/expression tasks)
 *
 * Subscribes to the pipeline SSE stream and renders the visual that
 * the running Python script tells it to show via @@EEG_EVENT lines.
 */

import { useState, useEffect, useCallback } from 'react'
import { X, Brain, CheckCircle, XCircle } from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────────────────

type VisualType = 'checkerboard' | 'breathing_circle' | 'word' | 'instruction'

interface TaskInfo {
  task: string
  visual: VisualType
  word?: string | null
  instruction: string
  duration: number
  category: string
  task_num: number
  total_tasks: number
}

interface TaskResult {
  task: string
  quality_passed: boolean
  quality_message?: string
  feature_dim?: number
}

interface EnrollmentSummary {
  success: boolean
  tasks_completed: number
  tasks_total: number
  feature_dim: number
  mean_similarity: number
  std_similarity: number
  threshold_1std: number
  threshold_2std: number
}

type Phase = 'connecting' | 'running' | 'training' | 'complete' | 'error'

// ── Visual Components (inline – small SVG / CSS) ────────────────────────────

function Checkerboard() {
  const size = 10
  const rects = []
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if ((r + c) % 2 === 0) {
        rects.push(
          <rect key={`${r}-${c}`} x={c * 10} y={r * 10} width={10} height={10} fill="white" />
        )
      }
    }
  }
  return (
    <svg viewBox="0 0 100 100" className="w-64 h-64 md:w-80 md:h-80 rounded-lg shadow-2xl">
      <rect width="100" height="100" fill="#1a1a2e" />
      {rects}
    </svg>
  )
}

function BreathingCircle({ duration }: { duration: number }) {
  // One full cycle = expand(4s) + hold(4s) + shrink(4s) = 12s
  const cycleTime = Math.min(12, duration)
  return (
    <div className="flex items-center justify-center w-64 h-64 md:w-80 md:h-80">
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <circle
          cx="50" cy="50" r="20"
          fill="none"
          stroke="#10b981"
          strokeWidth="2"
          opacity="0.3"
        />
        <circle
          cx="50" cy="50" r="20"
          fill="none"
          stroke="#10b981"
          strokeWidth="3"
          className="eeg-breathing-circle"
          style={{
            filter: 'drop-shadow(0 0 12px rgba(16,185,129,0.5))',
            animationDuration: `${cycleTime}s`,
          }}
        />
      </svg>
      <style>{`
        @keyframes breathe {
          0%   { r: 15; opacity: 0.6; }
          33%  { r: 38; opacity: 1;   }
          50%  { r: 38; opacity: 1;   }
          83%  { r: 15; opacity: 0.6; }
          100% { r: 15; opacity: 0.6; }
        }
        .eeg-breathing-circle {
          animation: breathe 12s ease-in-out infinite;
        }
      `}</style>
    </div>
  )
}

function WordDisplay({ word }: { word: string }) {
  return (
    <div className="flex items-center justify-center w-64 h-64 md:w-80 md:h-80 animate-fade-in">
      <span className="text-5xl md:text-6xl font-display font-bold text-white tracking-wider select-none">
        {word}
      </span>
    </div>
  )
}

function InstructionDisplay({ instruction }: { instruction: string }) {
  return (
    <div className="flex items-center justify-center w-64 h-64 md:w-80 md:h-80 px-6 animate-fade-in">
      <p className="text-xl md:text-2xl text-white/90 text-center font-medium leading-relaxed">
        {instruction}
      </p>
    </div>
  )
}

function ProgressBar({ elapsed, total }: { elapsed: number; total: number }) {
  // Smoothly interpolate between server-reported seconds using requestAnimationFrame
  const [displayPct, setDisplayPct] = useState(0)
  const targetPct = total > 0 ? Math.min(100, (elapsed / total) * 100) : 0
  const remaining = Math.max(0, Math.ceil(total - elapsed))

  useEffect(() => {
    // When we get a new target, animate smoothly toward it
    let raf: number
    const animate = () => {
      setDisplayPct(prev => {
        const diff = targetPct - prev
        if (Math.abs(diff) < 0.5) return targetPct
        // Ease toward target — covers the gap in ~300ms
        return prev + diff * 0.15
      })
      raf = requestAnimationFrame(animate)
    }
    raf = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(raf)
  }, [targetPct])

  return (
    <div className="w-full max-w-sm">
      <div className="flex justify-between text-xs text-white/60 mb-1">
        <span>Recording</span>
        <span>{remaining}s</span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full bg-emerald-500 rounded-full"
          style={{ width: `${displayPct}%` }}
        />
      </div>
    </div>
  )
}

// ── Main Modal ───────────────────────────────────────────────────────────────

interface EegEnrollmentModalProps {
  onClose: () => void
  authenticatedFetch: (url: string, init?: RequestInit) => Promise<Response>
}

export function EegEnrollmentModal({ onClose, authenticatedFetch }: EegEnrollmentModalProps) {
  const [phase, setPhase] = useState<Phase>('connecting')
  const [currentTask, setCurrentTask] = useState<TaskInfo | null>(null)
  const [progress, setProgress] = useState({ elapsed: 0, total: 0 })
  const [completedTasks, setCompletedTasks] = useState<TaskResult[]>([])
  const [summary, setSummary] = useState<EnrollmentSummary | null>(null)
  // Parse @@EEG_EVENT lines from SSE data frames
  const handleSseMessage = useCallback((raw: string) => {
    // Each SSE data frame is JSON with { line, index } or { event: "done" }
    let frame: any
    try {
      frame = JSON.parse(raw)
    } catch {
      return
    }

    // Stream-level done
    if (frame.event === 'done') {
      setPhase('complete')
      return
    }

    const line: string = frame.line || ''
    const marker = '@@EEG_EVENT:'
    const idx = line.indexOf(marker)
    if (idx === -1) return

    let evt: any
    try {
      evt = JSON.parse(line.slice(idx + marker.length))
    } catch {
      return
    }

    switch (evt.event) {
      case 'enrollment_start':
        // Don't transition yet — stay on "Connecting" until the first task arrives
        break

      case 'task_start':
        // Set current task first, THEN transition phase so both render together
        setCurrentTask({
          task: evt.task,
          visual: evt.visual,
          word: evt.word,
          instruction: evt.instruction,
          duration: evt.duration,
          category: evt.category || '',
          task_num: evt.task_num,
          total_tasks: evt.total_tasks,
        })
        setProgress({ elapsed: 0, total: evt.duration })
        setPhase('running')
        break

      case 'recording_progress':
        setProgress({ elapsed: evt.elapsed, total: evt.total })
        break

      case 'task_complete':
        setCompletedTasks(prev => [...prev, {
          task: evt.task,
          quality_passed: evt.quality_passed,
          quality_message: evt.quality_message,
          feature_dim: evt.feature_dim,
        }])
        break

      case 'enrollment_complete':
        setSummary({
          success: evt.success,
          tasks_completed: evt.tasks_completed,
          tasks_total: evt.tasks_total,
          feature_dim: evt.feature_dim,
          mean_similarity: evt.mean_similarity,
          std_similarity: evt.std_similarity,
          threshold_1std: evt.threshold_1std,
          threshold_2std: evt.threshold_2std,
        })
        setPhase('complete')
        break
    }
  }, [])

  // Start the script and poll for output
  // (SSE / fetch streaming doesn't work through Vite's dev proxy — it buffers / caches)
  useEffect(() => {
    let cancelled = false

    // Fire-and-forget: start the script (backend deduplicates if already running)
    authenticatedFetch('/api/mcp/pipeline.run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        script: 'enroll_brainwaves',
        args: ['--dashboard'],
      }),
    }).catch(() => {
      /* response arrives when script finishes – we don't need it */
    })

    // Poll the output endpoint every 200ms
    const poll = async () => {
      let cursor = 0

      while (!cancelled) {
        try {
          const res = await fetch(`/api/mcp/pipeline.output/enroll_brainwaves?cursor=${cursor}`)
          const data = await res.json()

          // Script hasn't registered yet — keep waiting
          if (!data.started && !data.done) {
            await new Promise(r => setTimeout(r, 300))
            continue
          }

          for (const { line, index } of data.lines) {
            handleSseMessage(JSON.stringify({ line, index }))
            cursor = index + 1
          }

          if (data.done) {
            handleSseMessage(JSON.stringify({ event: 'done', exitCode: data.exitCode }))
            break
          }
        } catch {
          // Network hiccup — retry on next tick
        }
        await new Promise(r => setTimeout(r, 100))
      }
    }
    poll()

    return () => { cancelled = true }
  }, [authenticatedFetch, handleSseMessage])

  // ── Render ─────────────────────────────────────────────────────────────────

  const renderVisual = () => {
    if (!currentTask) return null
    switch (currentTask.visual) {
      case 'checkerboard':
        return <Checkerboard />
      case 'breathing_circle':
        return <BreathingCircle duration={currentTask.duration} />
      case 'word':
        return <WordDisplay word={currentTask.word || '...'} />
      case 'instruction':
        return <InstructionDisplay instruction={currentTask.instruction} />
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col items-center justify-center">
      {/* Close button */}
      <button
        onClick={onClose}
        className="absolute top-6 right-6 p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white/70 hover:text-white z-10"
      >
        <X className="w-5 h-5" />
      </button>

      {/* ── Connecting / Waiting for first task ── */}
      {phase === 'connecting' && (
        <div className="flex flex-col items-center gap-6 animate-fade-in">
          <Brain className="w-16 h-16 text-emerald-400 animate-pulse" />
          <h2 className="text-2xl font-display font-semibold text-white">Connecting to EEG Device</h2>
          <p className="text-white/60 text-sm">Preparing enrollment session...</p>
          <div className="w-8 h-8 border-2 border-white/20 border-t-emerald-400 rounded-full animate-spin mt-2" />
        </div>
      )}

      {/* ── Running tasks ── */}
      {phase === 'running' && currentTask && (
        <div className="flex flex-col items-center gap-8 w-full max-w-lg px-6 animate-fade-in">
          {/* Task counter */}
          <div className="w-full">
            <div className="flex justify-between text-xs text-white/50 mb-1.5">
              <span>Task {currentTask.task_num} of {currentTask.total_tasks}</span>
              <span className="capitalize">{currentTask.category}</span>
            </div>
            <div className="h-1 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500/60 rounded-full transition-all duration-500"
                style={{ width: `${(currentTask.task_num / currentTask.total_tasks) * 100}%` }}
              />
            </div>
          </div>

          {/* Task title */}
          <h2 className="text-lg font-display font-semibold text-white/80 uppercase tracking-widest">
            {currentTask.task.replace(/_/g, ' ')}
          </h2>

          {/* Visual element */}
          <div className="flex items-center justify-center">
            {renderVisual()}
          </div>

          {/* Instruction (for non-word/instruction visuals show separate text) */}
          {(currentTask.visual === 'checkerboard' || currentTask.visual === 'breathing_circle') && (
            <p className="text-sm text-white/50 text-center max-w-md leading-relaxed">
              {currentTask.instruction}
            </p>
          )}

          {/* Recording progress */}
          <ProgressBar elapsed={progress.elapsed} total={progress.total} />
        </div>
      )}

      {/* ── Complete ── */}
      {phase === 'complete' && (
        <div className="flex flex-col items-center gap-8 w-full max-w-lg px-6 animate-fade-in">
          <div className="flex items-center gap-3">
            <CheckCircle className="w-10 h-10 text-emerald-400" />
            <h2 className="text-2xl font-display font-semibold text-white">Enrollment Complete</h2>
          </div>

          {/* Summary cards */}
          {summary && (
            <div className="grid grid-cols-2 gap-4 w-full">
              <div className="p-4 rounded-lg bg-white/5 border border-emerald-500/20 text-center">
                <div className="text-xs text-white/50 mb-1">Tasks Passed</div>
                <div className="text-2xl font-bold text-white">{summary.tasks_completed}/{summary.tasks_total}</div>
              </div>
              <div className="p-4 rounded-lg bg-white/5 border border-emerald-500/20 text-center">
                <div className="text-xs text-white/50 mb-1">Feature Dimensions</div>
                <div className="text-2xl font-bold text-white">{summary.feature_dim}</div>
              </div>
              <div className="p-4 rounded-lg bg-white/5 border border-emerald-500/20 text-center">
                <div className="text-xs text-white/50 mb-1">Mean Similarity</div>
                <div className="text-2xl font-bold text-white">{summary.mean_similarity.toFixed(4)}</div>
              </div>
              <div className="p-4 rounded-lg bg-white/5 border border-emerald-500/20 text-center">
                <div className="text-xs text-white/50 mb-1">1{'\u03C3'} Threshold</div>
                <div className="text-2xl font-bold text-white">{summary.threshold_1std.toFixed(4)}</div>
              </div>
            </div>
          )}

          {/* Task results */}
          <div className="w-full space-y-2">
            <h3 className="text-xs text-white/40 uppercase tracking-wider">Task Results</h3>
            <div className="flex flex-wrap gap-2">
              {completedTasks.map((t, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                    t.quality_passed
                      ? 'bg-emerald-500/15 text-emerald-400'
                      : 'bg-red-500/15 text-red-400'
                  }`}
                >
                  {t.quality_passed ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                  {t.task.replace(/_/g, ' ')}
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={onClose}
            className="px-8 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors"
          >
            Done
          </button>
        </div>
      )}

      {/* ── Error ── */}
      {phase === 'error' && (
        <div className="flex flex-col items-center gap-6 animate-fade-in">
          <XCircle className="w-12 h-12 text-red-400" />
          <h2 className="text-xl font-display font-semibold text-white">Enrollment Failed</h2>
          <p className="text-white/60 text-sm">An unexpected error occurred.</p>
          <button
            onClick={onClose}
            className="px-6 py-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-colors"
          >
            Close
          </button>
        </div>
      )}

      {/* Fade-in keyframe (if not already in global styles) */}
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fadeIn 0.4s ease-out; }
      `}</style>
    </div>
  )
}
