import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'

interface CollapsibleSectionProps {
  title: string
  icon?: React.ComponentType<{ className?: string }>
  children: React.ReactNode
  defaultExpanded?: boolean
  detailsContent?: React.ReactNode
}

export function CollapsibleSection({ 
  title, 
  icon: Icon, 
  children, 
  defaultExpanded = false,
  detailsContent 
}: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [showDetails, setShowDetails] = useState(false)
  
  return (
    <div className="card">
      <div 
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className="font-display font-semibold text-text-primary flex items-center gap-2">
          {Icon && <Icon className="w-5 h-5 text-accent" />}
          {title}
        </h3>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-text-muted" />
        ) : (
          <ChevronDown className="w-5 h-5 text-text-muted" />
        )}
      </div>
      {isExpanded && (
        <div className="mt-4">
          {children}
          {detailsContent && (
            <div className="mt-4 pt-4 border-t border-surface-200">
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setShowDetails(!showDetails)
                }}
                className="text-sm text-accent hover:text-accent/80 flex items-center gap-1"
              >
                {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                {showDetails ? 'Hide' : 'Show'} Details
              </button>
              {showDetails && (
                <div className="mt-2">
                  {detailsContent}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

