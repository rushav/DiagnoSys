'use client'

import { useRouter, useSearchParams, usePathname } from 'next/navigation'
import { useCallback } from 'react'
import type { ProblemsFilters } from '@/lib/api'

const DOMAINS = [
  'databases', 'backend', 'frontend', 'ml', 'devops', 'security',
  'mobile', 'systems', 'networking', 'web', 'cloud', 'testing',
  'performance', 'architecture', 'data_engineering', 'nlp',
  'computer_vision', 'rl', 'robotics', 'other',
]

const STATUS_OPTIONS = [
  { value: '', label: 'All' },
  { value: 'unsolved', label: 'Unsolved' },
  { value: 'partial', label: 'Partial' },
  { value: 'adequate', label: 'Adequate' },
]

interface FilterPanelProps {
  filters: ProblemsFilters
  onChange: (filters: Partial<ProblemsFilters>) => void
}

export function FilterPanel({ filters, onChange }: FilterPanelProps) {
  return (
    <aside className="w-full md:w-64 space-y-6 shrink-0">
      {/* Domain filter */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Domain</h3>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => onChange({ domain: undefined, page: 1 })}
            className={`px-2 py-1 text-xs rounded-full border transition-colors ${
              !filters.domain
                ? 'bg-blue-500 text-white border-blue-500'
                : 'bg-white text-gray-600 border-gray-200 hover:border-blue-300'
            }`}
          >
            All
          </button>
          {DOMAINS.map((d) => (
            <button
              key={d}
              onClick={() => onChange({ domain: d === filters.domain ? undefined : d, page: 1 })}
              className={`px-2 py-1 text-xs rounded-full border transition-colors ${
                filters.domain === d
                  ? 'bg-blue-500 text-white border-blue-500'
                  : 'bg-white text-gray-600 border-gray-200 hover:border-blue-300'
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* Status filter */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Solution Status</h3>
        <div className="flex flex-col gap-1">
          {STATUS_OPTIONS.map((opt) => (
            <label key={opt.value} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="status"
                value={opt.value}
                checked={(filters.status ?? '') === opt.value}
                onChange={() => onChange({ status: (opt.value as any) || undefined, page: 1 })}
                className="accent-blue-500"
              />
              <span className="text-sm text-gray-700">{opt.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Min confidence */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          Min Confidence: {((filters.min_confidence ?? 0) * 100).toFixed(0)}%
        </h3>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={filters.min_confidence ?? 0}
          onChange={(e) => onChange({ min_confidence: parseFloat(e.target.value), page: 1 })}
          className="w-full accent-blue-500"
        />
      </div>
    </aside>
  )
}
