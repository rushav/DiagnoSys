import Link from 'next/link'
import { ExternalLink, Clock } from 'lucide-react'
import type { Problem } from '@/lib/api'

const STATUS_STYLES: Record<string, string> = {
  unsolved: 'bg-red-100 text-red-700',
  partial: 'bg-yellow-100 text-yellow-700',
  adequate: 'bg-green-100 text-green-700',
}

const SOURCE_LABELS: Record<string, string> = {
  stack_exchange: 'Stack Exchange',
  github: 'GitHub',
  reddit: 'Reddit',
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

interface ProblemCardProps {
  problem: Problem
}

export function ProblemCard({ problem }: ProblemCardProps) {
  const status = problem.solution_status ?? 'unsolved'
  return (
    <Link
      href={`/problems/${problem.id}`}
      className="block bg-white rounded-xl border border-gray-100 shadow-sm hover:shadow-md hover:border-blue-200 transition-all p-5"
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <h3 className="text-sm font-semibold text-gray-900 line-clamp-2 flex-1">
          {problem.title}
        </h3>
        <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-600'}`}>
          {status}
        </span>
      </div>

      {problem.description && (
        <p className="text-xs text-gray-500 line-clamp-2 mb-3">
          {problem.description.replace(/<[^>]+>/g, '')}
        </p>
      )}

      <div className="flex flex-wrap gap-1 mb-3">
        {problem.domains.slice(0, 4).map((d) => (
          <span key={d.name} className="text-xs bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
            {d.name}
            {d.confidence !== null ? ` ${(d.confidence * 100).toFixed(0)}%` : ''}
          </span>
        ))}
      </div>

      <div className="flex items-center justify-between text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <ExternalLink className="h-3 w-3" />
          {SOURCE_LABELS[problem.source] ?? problem.source}
        </span>
        <span className="flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {timeAgo(problem.created_at)}
        </span>
        {problem.confidence_score !== null && (
          <span>{(problem.confidence_score * 100).toFixed(0)}% confidence</span>
        )}
      </div>
    </Link>
  )
}
