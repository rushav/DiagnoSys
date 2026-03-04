'use client'

import { useEffect, useRef, useState } from 'react'
import { Search, X, Loader2 } from 'lucide-react'

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  isLoading?: boolean
}

export function SearchBar({
  value,
  onChange,
  placeholder = 'Search engineering problems...',
  isLoading = false,
}: SearchBarProps) {
  const [localValue, setLocalValue] = useState(value)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    setLocalValue(value)
  }, [value])

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const next = e.target.value
    setLocalValue(next)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onChange(next), 300)
  }

  function handleClear() {
    setLocalValue('')
    onChange('')
  }

  return (
    <div className="relative w-full">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
      <input
        type="text"
        value={localValue}
        onChange={handleChange}
        placeholder={placeholder}
        className="w-full pl-10 pr-10 py-3 rounded-xl border border-gray-200 bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 placeholder-gray-400"
      />
      <div className="absolute right-3 top-1/2 -translate-y-1/2">
        {isLoading ? (
          <Loader2 className="h-5 w-5 text-gray-400 animate-spin" />
        ) : localValue ? (
          <button onClick={handleClear} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        ) : null}
      </div>
    </div>
  )
}
