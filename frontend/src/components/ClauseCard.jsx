import Tilt from 'react-parallax-tilt'
import { motion } from 'framer-motion'

const RISK_CONFIG = {
  high:   { color: 'var(--risk-high)',   label: 'HIGH' },
  medium: { color: 'var(--risk-medium)', label: 'MEDIUM' },
  low:    { color: 'var(--risk-low)',    label: 'LOW' },
}

export default function ClauseCard({ clause, index }) {
  const risk = RISK_CONFIG[clause.risk_level] ?? RISK_CONFIG.low

  return (
    <motion.div
      initial={{ opacity: 0, y: 48 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{
        duration: 0.65,
        delay: Math.min(index * 0.07, 0.35),
        ease: [0.16, 1, 0.3, 1],
      }}
    >
      <Tilt
        tiltMaxAngleX={4}
        tiltMaxAngleY={4}
        glareEnable={false}
        transitionSpeed={2200}
        scale={1.01}
      >
        <div
          className={`clause-card ${clause.risk_level}`}
          style={{ '--card-risk-color': risk.color }}
        >
          <div className="clause-card-header">
            <h3 className="clause-title">{clause.title}</h3>
            <span className={`risk-badge ${clause.risk_level}`}>{risk.label}</span>
          </div>

          <p className="clause-simplified">{clause.simplified}</p>

          {clause.risk_reason && (
            <div className="clause-risk-reason">
              <span className="risk-reason-icon">⚠</span>
              <span>{clause.risk_reason}</span>
            </div>
          )}
        </div>
      </Tilt>
    </motion.div>
  )
}
