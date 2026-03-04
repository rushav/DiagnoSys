import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Providers from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'DiagnoSys — Discover Unsolved Engineering Problems',
  description: 'ML-powered platform cataloguing unsolved engineering problems from Stack Exchange, GitHub, and Reddit.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 text-gray-900 min-h-screen`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
