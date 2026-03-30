import { useEffect, useState } from 'react'
import Particles, { initParticlesEngine } from '@tsparticles/react'
import { loadSlim } from '@tsparticles/slim'

const OPTIONS = {
  background: { color: { value: 'transparent' } },
  fpsLimit: 60,
  particles: {
    number: { value: 55, density: { enable: true, area: 900 } },
    color: { value: '#C9A96E' },
    opacity: { value: { min: 0.04, max: 0.18 } },
    size: { value: { min: 1, max: 2.5 } },
    links: {
      enable: true,
      color: '#C9A96E',
      opacity: 0.07,
      distance: 140,
      width: 1,
    },
    move: {
      enable: true,
      speed: 0.35,
      direction: 'none',
      random: true,
      outModes: { default: 'bounce' },
    },
  },
  interactivity: {
    events: {
      onHover: { enable: true, mode: 'repulse' },
    },
    modes: {
      repulse: { distance: 90, duration: 0.4 },
    },
  },
  detectRetina: true,
}

export default function ParticleBackground() {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine)
    }).then(() => setReady(true))
  }, [])

  if (!ready) return null

  return (
    <Particles
      id="tsparticles"
      options={OPTIONS}
      style={{ position: 'absolute', inset: 0, zIndex: 0 }}
    />
  )
}
