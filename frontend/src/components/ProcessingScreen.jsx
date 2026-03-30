import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const STEPS = [
  'Extracting document text',
  'Identifying legal clauses',
  'Assessing risk levels',
  'Finalizing results',
]

const STEP_DELAYS = [0, 1500, 3500, 6500]

export default function ProcessingScreen({ fileName }) {
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    const timers = STEP_DELAYS.slice(1).map((delay, i) =>
      setTimeout(() => setActiveStep(i + 1), delay)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <motion.div
      className="processing-screen"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      <motion.div
        className="processing-content"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
      >
        <div>
          <h2 className="processing-title">Analyzing your document</h2>
          {fileName && <p className="processing-filename">{fileName}</p>}
        </div>

        <div className="stepper">
          {STEPS.map((step, i) => {
            const state = i < activeStep ? 'done' : i === activeStep ? 'active' : 'idle'
            return (
              <motion.div
                key={step}
                className={`stepper-step ${state}`}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: state === 'idle' ? 0.3 : 1, x: 0 }}
                transition={{ duration: 0.5, delay: i * 0.12, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="stepper-dot">
                  {state === 'done' && '✓'}
                  {state === 'active' && <PulseDot />}
                </div>
                <span className="stepper-label">{step}</span>
              </motion.div>
            )
          })}
        </div>
      </motion.div>
    </motion.div>
  )
}

function PulseDot() {
  return (
    <motion.span
      className="pulse-dot"
      animate={{ scale: [1, 1.5, 1], opacity: [1, 0.4, 1] }}
      transition={{ duration: 1.1, repeat: Infinity, ease: 'easeInOut' }}
    />
  )
}
