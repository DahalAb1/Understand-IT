import { useEffect } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export default function CustomCursor() {
  const mouseX = useMotionValue(-100)
  const mouseY = useMotionValue(-100)

  const dotX = useSpring(mouseX, { damping: 28, stiffness: 500 })
  const dotY = useSpring(mouseY, { damping: 28, stiffness: 500 })

  const ringX = useSpring(mouseX, { damping: 40, stiffness: 180 })
  const ringY = useSpring(mouseY, { damping: 40, stiffness: 180 })

  useEffect(() => {
    if (window.matchMedia('(hover: none)').matches) return

    function onMove(e) {
      mouseX.set(e.clientX)
      mouseY.set(e.clientY)
    }

    window.addEventListener('mousemove', onMove)
    return () => window.removeEventListener('mousemove', onMove)
  }, [mouseX, mouseY])

  return (
    <>
      <motion.div
        className="cursor-dot"
        style={{ x: dotX, y: dotY, translateX: '-50%', translateY: '-50%' }}
      />
      <motion.div
        className="cursor-ring"
        style={{ x: ringX, y: ringY, translateX: '-50%', translateY: '-50%' }}
      />
    </>
  )
}
