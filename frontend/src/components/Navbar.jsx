import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    function onScroll() {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <motion.nav
      className={`navbar ${scrolled ? 'scrolled' : ''}`}
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="navbar-logo">UNDERSTAND IT</span>
      <div className="navbar-links">
        <a href="#upload" className="navbar-link">How it works</a>
        <a
          href="https://github.com/DahalAb1/Understand-IT"
          target="_blank"
          rel="noopener noreferrer"
          className="navbar-link"
        >
          GitHub ↗
        </a>
      </div>
    </motion.nav>
  )
}
