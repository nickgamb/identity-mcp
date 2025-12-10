import { useEffect, useRef } from 'react'
// @ts-ignore - wordcloud doesn't have proper types
import wordcloud from 'wordcloud'

interface WordCloudProps {
  words: Array<[string, number]>
  maxWords?: number
}

export function WordCloud({ words, maxWords = 100 }: WordCloudProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  useEffect(() => {
    if (!canvasRef.current || !words.length) return
    
    const sortedWords = words
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxWords)
      .map(([word, freq]) => [word, Math.max(freq * 1000, 10)])
    
    try {
      wordcloud(canvasRef.current, {
        list: sortedWords as any,
        gridSize: 8,
        weightFactor: 1,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        color: () => {
          const colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe']
          return colors[Math.floor(Math.random() * colors.length)]
        },
        rotateRatio: 0.3,
        rotationSteps: 2,
        backgroundColor: 'transparent'
      })
    } catch (error) {
      console.error('Word cloud error:', error)
    }
  }, [words, maxWords])
  
  return (
    <div className="w-full h-64 bg-surface-100 rounded-lg flex items-center justify-center">
      <canvas ref={canvasRef} width={800} height={256} className="max-w-full" />
    </div>
  )
}

