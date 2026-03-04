'use client'

import { use } from 'react'
import Link from 'next/link'
import { ExternalLink, ArrowLeft, Loader2, AlertCircle } from 'lucide-react'
import { useProblem } from '@/lib/api'

const STATUS_STYLES: Record<string, string> = {
  unsolved: 'bg-red-100 text-red-700 border-red-200',
  partial: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  adequate: 'bg-green-100 text-green-700 border-green-200',
}

interface PageProps {
  params: Promise<{ id: string }>
}

export default function ProblemDetailPage({ params }: PageProps) {
  const { id } = use(params)
  const { data: problem, isLoading, isError } = useProblem(id)

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (isError || !problem) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <AlertCircle className="h-10 w-10 text-red-400" />
        <p className="text-gray-600">Problem not found</p>
        <Link href="/" className="text-blue-500 hover:underline">← Back to search</Link>
      </div>
    )
  }

  const status = problem.solution_status ?? 'unsolved'

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-5xl mx-auto px-4 py-4">
          <Link href="/" className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-900 w-fit">
            <ArrowLeft className="h-4 w-4" /> Back to search
          </Link>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main content */}
          <div className="flex-1 min-w-0">
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
              <div className="flex items-start justify-between gap-4 mb-4">
                <h1 className="text-xl font-bold text-gray-900 flex-1">
                  {problem.title}
                </h1>
                <span className={`shrink-0 text-sm px-3 py-1 rounded-full font-medium border ${STATUS_STYLES[status]}`}>
                  {status}
                </span>
              </div>

              {/* Domain tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {problem.domains.map((d) => (
                  <span key={d.name} className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded-full font-medium">
                    {d.name}
                    {d.confidence !== null ? ` · ${(d.confidence * 100).toFixed(0)}%` : ''}
                  </span>
                ))}
              </div>

              {problem.confidence_score !== null && (
                <p className="text-sm text-gray-500 mb-4">
                  Overall confidence: <strong>{(problem.confidence_score * 100).toFixed(0)}%</strong>
                </p>
              )}

              {/* Description */}
              {problem.description && (
                <div className="prose prose-sm max-w-none text-gray-700 border-t border-gray-100 pt-4">
                  <p className="whitespace-pre-wrap">{problem.description.replace(/<[^>]+>/g, '')}</p>
                </div>
              )}

              {/* Source link */}
              <div className="mt-6 pt-4 border-t border-gray-100">
                <a
                  href={problem.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                >
                  <ExternalLink className="h-4 w-4" />
                  View on {problem.source === 'stack_exchange' ? 'Stack Exchange' : problem.source === 'github' ? 'GitHub' : 'Reddit'}
                </a>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <aside className="lg:w-72 shrink-0 space-y-4">
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Problem Details</h3>
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <dt className="text-gray-500">Source</dt>
                  <dd className="text-gray-900 font-medium capitalize">{problem.source.replace('_', ' ')}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Status</dt>
                  <dd className={`font-medium capitalize ${
                    status === 'unsolved' ? 'text-red-600' :
                    status === 'partial' ? 'text-yellow-600' : 'text-green-600'
                  }`}>{status}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Added</dt>
                  <dd className="text-gray-900">{new Date(problem.created_at).toLocaleDateString()}</dd>
                </div>
              </dl>
            </div>

            {/* Related problems placeholder */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-4">
              <h3 className="font-semibold text-gray-700 mb-2 text-sm">Related Problems</h3>
              <p className="text-xs text-gray-400">Semantic search coming soon (Pinecone/Weaviate integration)</p>
            </div>
          </aside>
        </div>
      </main>
    </div>
  )
}
