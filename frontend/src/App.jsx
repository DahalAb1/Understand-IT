import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import CustomCursor from './components/CustomCursor'
import UploadScreen from './components/UploadScreen'
import ProcessingScreen from './components/ProcessingScreen'
import ResultsScreen from './components/ResultsScreen'

export default function App() {
  const [screen, setScreen] = useState('upload')
  const [file, setFile] = useState(null)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  async function handleSimplify(selectedFile) {
    setFile(selectedFile)
    setScreen('processing')
    setError(null)

    try {
      const body = new FormData()
      body.append('file', selectedFile)

      const res = await fetch('http://localhost:8000/simplify', {
        method: 'POST',
        body,
      })

      if (!res.ok) {
        const msg = await res.json().catch(() => ({}))
        throw new Error(msg.detail ?? 'Something went wrong. Please try again.')
      }

      const data = await res.json()
      setResults(data)
      setScreen('results')
    } catch (err) {
      setError(err.message)
      setScreen('upload')
    }
  }

  function handleReset() {
    setScreen('upload')
    setFile(null)
    setResults(null)
    setError(null)
  }

  return (
    <>
      <CustomCursor />
      <AnimatePresence mode="wait">
        {screen === 'upload' && (
          <UploadScreen key="upload" onSimplify={handleSimplify} error={error} />
        )}
        {screen === 'processing' && (
          <ProcessingScreen key="processing" fileName={file?.name} />
        )}
        {screen === 'results' && (
          <ResultsScreen
            key="results"
            results={results}
            fileName={file?.name}
            onReset={handleReset}
          />
        )}
      </AnimatePresence>
    </>
  )
}
