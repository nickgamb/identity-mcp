/**
 * EEG Authorization Modal
 *
 * Full-screen live authorization experience:
 *  - Static enrolled spectral profile (reference shape)
 *  - Live EEG spectral overlay (shows match / divergence)
 *  - Real-time similarity gauge (circular arc)
 *  - Rolling score timeline chart
 *  - Stop button to terminate the script
 *
 * Subscribes to the pipeline SSE stream and updates on each
 * @@EEG_EVENT assurance_result from the running Python script.
 */

import { useState, useEffect, useCallback } from 'react'
import { X, Brain, Square, Activity } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, ReferenceLine,
} from 'recharts'

// ── Types ────────────────────────────────────────────────────────────────────

interface SpectralProfile {
  [channel: string]: {
    [band: string]: number | { mean_relative_power: number; std_relative_power?: number }
  }
}

interface AssuranceResult {
  window: number
  assurance_score: number
  similarity: number
  confidence: string
  live_spectral: SpectralProfile | null
  timestamp?: string
}

type Phase = 'connecting' | 'running' | 'stopped' | 'error'

const BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma'] as const

// ── Similarity Gauge (SVG arc) ───────────────────────────────────────────────

function SimilarityGauge({ score, confidence }: { score: number; confidence: string }) {
  const radius = 70
  const stroke = 10
  const circumference = Math.PI * radius // half-circle
  const filled = circumference * Math.min(1, Math.max(0, score))

  // Color based on score
  let color = '#ef4444' // red
  if (score >= 0.6) color = '#10b981'      // green
  else if (score >= 0.3) color = '#f59e0b' // amber

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 160 100" className="w-48 h-28">
        {/* Background arc */}
        <path
          d="M 10 90 A 70 70 0 0 1 150 90"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <path
          d="M 10 90 A 70 70 0 0 1 150 90"
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={circumference - filled}
          style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.4s ease' }}
        />
        {/* Score text */}
        <text x="80" y="78" textAnchor="middle" className="text-3xl font-bold" fill="white">
          {score.toFixed(2)}
        </text>
      </svg>
      <div className="text-xs text-white/50 uppercase tracking-wider mt-1">{confidence.replace(/_/g, ' ')}</div>
    </div>
  )
}

// ── Main Modal ───────────────────────────────────────────────────────────────

interface EegAuthorizationModalProps {
  onClose: () => void
  authenticatedFetch: (url: string, init?: RequestInit) => Promise<Response>
}

export function EegAuthorizationModal({ onClose, authenticatedFetch }: EegAuthorizationModalProps) {
  const [phase, setPhase] = useState<Phase>('connecting')
  const [enrolledSpectral, setEnrolledSpectral] = useState<SpectralProfile | null>(null)
  const [liveSpectral, setLiveSpectral] = useState<SpectralProfile | null>(null)
  const [currentScore, setCurrentScore] = useState(0)
  const [currentConfidence, setCurrentConfidence] = useState('--')
  const [scoreHistory, setScoreHistory] = useState<{ window: number; score: number }[]>([])
  const [threshold1std, setThreshold1std] = useState<number | null>(null)
  const [channels, setChannels] = useState<string[]>([])
  const [windowNum, setWindowNum] = useState(0)
  const [readingProgress, setReadingProgress] = useState({ elapsed: 0, total: 0 })
  // Normalize spectral values — they can be plain numbers or objects
  const getRelPower = (val: any): number => {
    if (typeof val === 'number') return val
    if (val && typeof val === 'object') return val.mean_relative_power ?? 0
    return 0
  }

  // Parse SSE frames
  const handleSseMessage = useCallback((raw: string) => {
    let frame: any
    try { frame = JSON.parse(raw) } catch { return }

    if (frame.event === 'done') {
      setPhase('stopped')
      return
    }
    if (frame.event === 'error') {
      setPhase('stopped')
      return
    }

    const line: string = frame.line || ''
    const marker = '@@EEG_EVENT:'
    const idx = line.indexOf(marker)
    if (idx === -1) return

    let evt: any
    try { evt = JSON.parse(line.slice(idx + marker.length)) } catch { return }

    switch (evt.event) {
      case 'auth_start': {
        const sp = evt.spectral_profile || {}
        setEnrolledSpectral(sp)
        setChannels(Object.keys(sp))
        const stats = evt.statistics || {}
        setThreshold1std(stats.similarity_threshold_1std ?? null)
        setPhase('running')
        break
      }

      case 'reading_start':
        setWindowNum(evt.window || 0)
        setReadingProgress({ elapsed: 0, total: evt.duration || 10 })
        break

      case 'reading_progress':
        setReadingProgress({ elapsed: evt.elapsed, total: evt.total })
        break

      case 'assurance_result': {
        const res = evt as AssuranceResult
        setCurrentScore(res.assurance_score)
        setCurrentConfidence(res.confidence)
        setLiveSpectral(res.live_spectral ?? null)
        setScoreHistory(prev => [...prev.slice(-29), { window: res.window, score: res.assurance_score }])
        break
      }

      case 'auth_stopped':
        setPhase('stopped')
        break
    }
  }, [])

  // Start script + poll for output
  // (SSE / fetch streaming doesn't work through Vite's dev proxy — it buffers / caches)
  useEffect(() => {
    let cancelled = false

    authenticatedFetch('/api/mcp/pipeline.run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        script: 'authorize_brainwaves',
        args: ['--dashboard', '--continuous', '--interval', '2', '--window', '5'],
      }),
    }).catch(() => { /* fires when script exits */ })

    const poll = async () => {
      let cursor = 0

      while (!cancelled) {
        try {
          const res = await fetch(`/api/mcp/pipeline.output/authorize_brainwaves?cursor=${cursor}`)
          const data = await res.json()

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
          // Network hiccup — retry
        }
        await new Promise(r => setTimeout(r, 100))
      }
    }
    poll()

    return () => { cancelled = true }
  }, [authenticatedFetch, handleSseMessage])

  const handleStop = async () => {
    try {
      await authenticatedFetch('/api/mcp/pipeline.stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script: 'authorize_brainwaves' }),
      })
    } catch { /* ignore */ }
    setPhase('stopped')
  }

  // Build spectral overlay chart data
  const spectralChartData = channels.map(ch => {
    const row: Record<string, any> = { channel: ch }
    BANDS.forEach(band => {
      const enrolled = enrolledSpectral?.[ch]?.[band]
      row[`enrolled_${band}`] = getRelPower(enrolled)
      const live = liveSpectral?.[ch]?.[band]
      row[`live_${band}`] = live != null ? getRelPower(live) : 0
    })
    // Compute total enrolled and live for the overlay comparison
    row.enrolled_total = BANDS.reduce((s, b) => s + (row[`enrolled_${b}`] || 0), 0)
    row.live_total = BANDS.reduce((s, b) => s + (row[`live_${b}`] || 0), 0)
    return row
  })

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-emerald-400" />
          <h2 className="text-lg font-display font-semibold text-white">Live EEG Authorization</h2>
          {phase === 'running' && (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Live
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {phase === 'running' && (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600/80 hover:bg-red-500 text-white text-sm font-medium transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Stop
            </button>
          )}
          <button
            onClick={onClose}
            className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors text-white/70 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-6">
        {phase === 'connecting' && (
          <div className="flex flex-col items-center justify-center h-full gap-6 animate-fade-in">
            <Brain className="w-16 h-16 text-emerald-400 animate-pulse" />
            <h2 className="text-xl font-display font-semibold text-white">Loading Model & Connecting</h2>
            <p className="text-white/50 text-sm">Preparing live authorization...</p>
          </div>
        )}

        {(phase === 'running' || phase === 'stopped') && (
          <div className="space-y-6 animate-fade-in">
            {/* Top row: Gauge + reading info */}
            <div className="flex flex-col md:flex-row items-center gap-6">
              <SimilarityGauge score={currentScore} confidence={currentConfidence} />
              <div className="flex-1 space-y-2">
                <div className="text-sm text-white/50">
                  Window #{windowNum}
                  {phase === 'running' && readingProgress.total > 0 && (
                    <span className="ml-3 text-white/30">
                      Reading: {readingProgress.elapsed.toFixed(0)}s / {readingProgress.total.toFixed(0)}s
                    </span>
                  )}
                </div>
                {/* Reading progress bar */}
                {phase === 'running' && readingProgress.total > 0 && (
                  <div className="h-1.5 bg-white/10 rounded-full overflow-hidden max-w-xs">
                    <div
                      className="h-full bg-cyan-500/70 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min(100, (readingProgress.elapsed / readingProgress.total) * 100)}%` }}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Spectral Overlay Chart */}
            {spectralChartData.length > 0 && (
              <div className="rounded-lg bg-white/5 border border-white/10 p-4">
                <h3 className="text-sm font-semibold text-white/70 mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  Spectral Overlay — Enrolled vs Live
                </h3>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spectralChartData} margin={{ top: 5, right: 20, left: 10, bottom: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="channel" stroke="rgba(255,255,255,0.4)" fontSize={9} angle={-45} textAnchor="end" height={60} />
                      <YAxis stroke="rgba(255,255,255,0.4)" fontSize={10} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.15)', fontSize: 11 }}
                        labelStyle={{ color: '#fff' }}
                      />
                      {/* Enrolled as semi-transparent bars */}
                      <Bar dataKey="enrolled_total" fill="#10b981" fillOpacity={0.35} name="Enrolled" />
                      {/* Live as opaque overlay */}
                      <Bar dataKey="live_total" fill="#06b6d4" fillOpacity={0.8} name="Live" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex gap-4 mt-2 text-xs text-white/40">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-emerald-500/40" /> Enrolled Profile</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-cyan-500" /> Live EEG</span>
                </div>
              </div>
            )}

            {/* Score Timeline */}
            {scoreHistory.length > 1 && (
              <div className="rounded-lg bg-white/5 border border-white/10 p-4">
                <h3 className="text-sm font-semibold text-white/70 mb-3">Assurance Score Over Time</h3>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={scoreHistory} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="window" stroke="rgba(255,255,255,0.4)" fontSize={10} label={{ value: 'Window', position: 'insideBottom', offset: -2, fill: 'rgba(255,255,255,0.4)', fontSize: 10 }} />
                      <YAxis domain={[0, 1]} stroke="rgba(255,255,255,0.4)" fontSize={10} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.15)', fontSize: 11 }}
                        labelStyle={{ color: '#fff' }}
                        formatter={(v: any) => [Number(v).toFixed(4), 'Score']}
                      />
                      <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981', r: 3 }} />
                      {threshold1std != null && (
                        <ReferenceLine y={threshold1std} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: '1\u03C3', position: 'right', fill: '#f59e0b', fontSize: 10 }} />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Stopped banner */}
            {phase === 'stopped' && (
              <div className="flex flex-col items-center gap-4 py-6">
                <p className="text-white/50 text-sm">Authorization session ended.</p>
                <button
                  onClick={onClose}
                  className="px-8 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors"
                >
                  Done
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fadeIn 0.4s ease-out; }
      `}</style>
    </div>
  )
}
