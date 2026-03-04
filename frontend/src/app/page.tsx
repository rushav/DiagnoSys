'use client'

import { useState } from 'react'
import { useProblems, type ProblemsFilters } from '@/lib/api'
import { ProblemCard } from '@/components/ProblemCard'
import { SearchBar } from '@/components/SearchBar'
import { FilterPanel } from '@/components/FilterPanel'
import { ChevronLeft, ChevronRight, AlertCircle, RefreshCw } from 'lucide-react'

function SkeletonCard() {
  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-5 animate-pulse">
      <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
      <div className="h-3 bg-gray-200 rounded w-full mb-1" />
      <div className="h-3 bg-gray-200 rounded w-2/3 mb-3" />
      <div className="flex gap-1 mb-3">
        {[1,2,3].map(i => <div key={i} className="h-5 bg-gray-200 rounded-full w-16" />)}
      </div>
      <div className="flex justify-between">
        <div className="h-3 bg-gray-200 rounded w-20" />
        <div className="h-3 bg-gray-200 rounded w-16" />
      </div>
    </div>
  )
}

export default function HomePage() {
  const [filters, setFilters] = useState<ProblemsFilters>({ page: 1, page_size: 20 })
  const [search, setSearch] = useState('')
  const { data, isLoading, isError, refetch } = useProblems(filters)

  function updateFilters(patch: Partial<ProblemsFilters>) {
    setFilters(prev => ({ ...prev, ...patch }))
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-100 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
          <h1 className="text-xl font-bold text-blue-600 shrink-0">DiagnoSys</h1>
          <SearchBar
            value={search}
            onChange={(v) => { setSearch(v); updateFilters({ page: 1 }) }}
            isLoading={isLoading}
          />
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row gap-8">
          {/* Filters */}
          <FilterPanel filters={filters} onChange={updateFilters} />

          {/* Results */}
          <div className="flex-1 min-w-0">
            {/* Stats */}
            {data && (
              <p className="text-sm text-gray-500 mb-4">
                {data.total.toLocaleString()} problems found
              </p>
            )}

            {/* Error */}
            {isError && (
              <div className="flex flex-col items-center justify-center py-16 gap-4">
                <AlertCircle className="h-10 w-10 text-red-400" />
                <p className="text-gray-600">Failed to load problems</p>
                <button
                  onClick={() => refetch()}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  <RefreshCw className="h-4 w-4" /> Retry
                </button>
              </div>
            )}

            {/* Loading skeletons */}
            {isLoading && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {Array.from({ length: 9 }).map((_, i) => <SkeletonCard key={i} />)}
              </div>
            )}

            {/* Problem cards */}
            {!isLoading && !isError && data && (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {data.items.map((p) => (
                    <ProblemCard key={p.id} problem={p} />
                  ))}
                </div>

                {data.items.length === 0 && (
                  <div className="text-center py-16 text-gray-500">
                    No problems match your filters.
                  </div>
                )}

                {/* Pagination */}
                {data.pages > 1 && (
                  <div className="flex items-center justify-center gap-4 mt-8">
                    <button
                      disabled={data.page <= 1}
                      onClick={() => updateFilters({ page: data.page - 1 })}
                      className="p-2 rounded-lg border disabled:opacity-40 hover:bg-gray-50"
                    >
                      <ChevronLeft className="h-5 w-5" />
                    </button>
                    <span className="text-sm text-gray-600">
                      Page {data.page} of {data.pages}
                    </span>
                    <button
                      disabled={data.page >= data.pages}
                      onClick={() => updateFilters({ page: data.page + 1 })}
                      className="p-2 rounded-lg border disabled:opacity-40 hover:bg-gray-50"
                    >
                      <ChevronRight className="h-5 w-5" />
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
