import { Search } from 'lucide-react'

interface SearchInputProps {
  placeholder: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  className?: string
}

export function SearchInput({ 
  placeholder, 
  value, 
  onChange,
  className = ""
}: SearchInputProps) {
  return (
    <div className={`relative flex-1 ${className}`}>
      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-muted" />
      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        className="w-full pl-10 pr-4 py-2 rounded-lg border border-surface-200 bg-surface-50 focus:border-accent focus:ring-2 focus:ring-accent/20 outline-none"
      />
    </div>
  )
}

