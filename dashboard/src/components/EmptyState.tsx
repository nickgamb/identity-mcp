interface EmptyStateProps {
  icon: React.ComponentType<{ className?: string }>
  title: string
  message: string
}

export function EmptyState({ 
  icon: Icon, 
  title, 
  message 
}: EmptyStateProps) {
  return (
    <div className="text-center py-12 text-text-muted">
      <Icon className="w-12 h-12 mx-auto mb-3 opacity-30" />
      <p className="font-medium mb-2">{title}</p>
      <p className="text-sm">{message}</p>
    </div>
  )
}

