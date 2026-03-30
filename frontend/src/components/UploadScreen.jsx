import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import Navbar from './Navbar'
import ParticleBackground from './ParticleBackground'

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadScreen({ onSimplify, error }) {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  function handleFile(f) {
    if (f?.type === 'application/pdf') setFile(f)
  }

  function onDragOver(e) {
    e.preventDefault()
    setDragging(true)
  }

  function onDragLeave(e) {
    if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <motion.div
      className="upload-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <ParticleBackground />
      <Navbar />

      <div className="upload-content" id="upload">
        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        >
          <h1 className="upload-headline">Understand IT</h1>
          <p className="upload-subline">Legal documents, decoded. No lawyer required.</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          style={{ width: '100%' }}
        >
          <div
            className={`upload-zone ${dragging ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => !file && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && !file && inputRef.current?.click()}
            aria-label="Upload PDF document"
          >
            <input
              ref={inputRef}
              type="file"
              accept=".pdf"
              className="sr-only"
              onChange={(e) => handleFile(e.target.files[0])}
            />

            {file ? (
              <div className="file-info">
                <span className="file-check">✓</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">{formatSize(file.size)}</span>
                <button
                  className="file-remove"
                  onClick={(e) => { e.stopPropagation(); setFile(null) }}
                  aria-label="Remove file"
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="upload-zone-inner">
                <div className="upload-icon">↑</div>
                <p className="upload-zone-text">Drag your PDF here</p>
                <p className="upload-zone-hint">or click to browse</p>
              </div>
            )}
          </div>
        </motion.div>

        {error && (
          <motion.p
            className="upload-error"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
          </motion.p>
        )}

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.45, ease: [0.16, 1, 0.3, 1] }}
        >
          <button
            className="btn btn-primary upload-btn"
            disabled={!file}
            onClick={() => file && onSimplify(file)}
          >
            Simplify Document →
          </button>
        </motion.div>
      </div>
    </motion.div>
  )
}
