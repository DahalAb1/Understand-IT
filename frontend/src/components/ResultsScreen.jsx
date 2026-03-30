import { useState } from 'react'
import { motion } from 'framer-motion'
import ClauseCard from './ClauseCard'

export default function ResultsScreen({ results, fileName, onReset }) {
  const [copied, setCopied] = useState(false)
  const clauses = results?.clauses ?? []

  const riskCounts = clauses.reduce((acc, c) => {
    acc[c.risk_level] = (acc[c.risk_level] ?? 0) + 1
    return acc
  }, {})

  function buildPlainText() {
    return clauses
      .map((c) =>
        [
          c.title,
          `Risk: ${c.risk_level.toUpperCase()}`,
          '',
          c.simplified,
          c.risk_reason ? `\nNote: ${c.risk_reason}` : '',
        ]
          .filter(Boolean)
          .join('\n')
      )
      .join('\n\n---\n\n')
  }

  function handleCopy() {
    navigator.clipboard.writeText(buildPlainText())
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleDownload() {
    const blob = new Blob([buildPlainText()], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = fileName?.replace('.pdf', '_simplified.txt') ?? 'simplified.txt'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <motion.div
      className="results-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="results-sticky-bar">
        <span className="results-bar-logo">UNDERSTAND IT</span>

        <div className="results-bar-meta">
          {fileName && <span className="results-filename">{fileName}</span>}
          <div className="risk-summary">
            {riskCounts.high > 0 && (
              <span className="risk-pill high">{riskCounts.high} High</span>
            )}
            {riskCounts.medium > 0 && (
              <span className="risk-pill medium">{riskCounts.medium} Medium</span>
            )}
            {riskCounts.low > 0 && (
              <span className="risk-pill low">{riskCounts.low} Low</span>
            )}
          </div>
        </div>

        <div className="results-bar-actions">
          <button className="btn btn-ghost" onClick={handleCopy}>
            {copied ? 'Copied!' : 'Copy All'}
          </button>
          <button className="btn btn-ghost" onClick={handleDownload}>
            Download
          </button>
          <button className="btn btn-primary" onClick={onReset}>
            New Document
          </button>
        </div>
      </div>

      <div className="results-content">
        <motion.div
          className="results-header"
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <h1 className="results-title">Document Analysis</h1>
          <p className="results-subtitle">
            {clauses.length} clause{clauses.length !== 1 ? 's' : ''} identified
          </p>
        </motion.div>

        <div className="results-grid">
          {clauses.map((clause, i) => (
            <ClauseCard key={i} clause={clause} index={i} />
          ))}
        </div>
      </div>
    </motion.div>
  )
}
