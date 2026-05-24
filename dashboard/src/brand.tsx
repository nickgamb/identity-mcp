/** Product branding (logo: universal-agent/static/favicon.png) */
export const BRAND = {
  company: 'MindGarden',
  logoSrc: '/mindgarden-logo.png',
  title: 'MINDGARDEN AI — identity dashboard',
  tagline: 'identity dashboard',
} as const

/** Header wordmark: MIND (normal) + GARDEN (bold) + AI™ */
export function BrandWordmark({ className = '' }: { className?: string }) {
  return (
    <span className={`font-display uppercase tracking-wide ${className}`}>
      <span className="font-normal">MIND</span>
      <span className="font-bold">GARDEN</span>
      <span className="font-normal"> AI™</span>
    </span>
  )
}
