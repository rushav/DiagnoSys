/**
 * api.ts — DiagnoSys type-safe API client
 * All types match backend Pydantic schemas exactly.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ── Types ──────────────────────────────────────────────────────────────────

export interface Domain {
  name: string
  confidence: number | null
}

export interface Problem {
  id: string
  title: string
  description: string | null
  source: 'stack_exchange' | 'github' | 'reddit'
  source_url: string
  solution_status: 'unsolved' | 'partial' | 'adequate' | null
  confidence_score: number | null
  domains: Domain[]
  created_at: string
  updated_at: string
}

export interface ProblemListResponse {
  items: Problem[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface ClassifyRequest {
  text: string
  context?: string
}

export interface ClassifyResponse {
  domains: string[]
  confidence: number
  model_version: string
}

export interface ProblemsFilters {
  page?: number
  page_size?: number
  domain?: string
  status?: 'unsolved' | 'partial' | 'adequate'
  min_confidence?: number
}

// ── API functions ──────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export function buildProblemsUrl(filters: ProblemsFilters): string {
  const params = new URLSearchParams()
  if (filters.page) params.set('page', String(filters.page))
  if (filters.page_size) params.set('page_size', String(filters.page_size))
  if (filters.domain) params.set('domain', filters.domain)
  if (filters.status) params.set('status', filters.status)
  if (filters.min_confidence !== undefined) params.set('min_confidence', String(filters.min_confidence))
  return `/api/problems?${params.toString()}`
}

export const api = {
  problems: {
    list: (filters: ProblemsFilters = {}) =>
      apiFetch<ProblemListResponse>(buildProblemsUrl(filters)),
    get: (id: string) =>
      apiFetch<Problem>(`/api/problems/${id}`),
  },
  ml: {
    classify: (req: ClassifyRequest) =>
      apiFetch<ClassifyResponse>('/ml/classify', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  },
}

// ── React Query hooks ──────────────────────────────────────────────────────

import { useQuery } from '@tanstack/react-query'

export function useProblems(filters: ProblemsFilters = {}) {
  return useQuery({
    queryKey: ['problems', filters],
    queryFn: () => api.problems.list(filters),
    staleTime: 60_000,
  })
}

export function useProblem(id: string) {
  return useQuery({
    queryKey: ['problem', id],
    queryFn: () => api.problems.get(id),
    enabled: !!id,
    staleTime: 120_000,
  })
}
