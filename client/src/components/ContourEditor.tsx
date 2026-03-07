import { useRef, useEffect, useCallback, useState } from 'react'
import type { Point } from '../lib/compare'

const HANDLE_R = 8
const MIN_POINTS = 3

interface ContourEditorProps {
  imageUrl: string
  imageSize: { width: number; height: number }
  points: Point[]
  onChange: (points: Point[]) => void
  paddingPx: number
  disabled?: boolean
}

export function ContourEditor({
  imageUrl,
  imageSize,
  points,
  onChange,
  paddingPx,
  disabled,
}: ContourEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 })
  const scaleRef = useRef({ scaleX: 1, scaleY: 1 })

  const updateDisplaySize = useCallback(() => {
    const img = imgRef.current
    const cont = containerRef.current
    if (!img || !cont) return
    // Размер и масштаб берём по реальному отображению картинки, а не по контейнеру
    const w = img.offsetWidth
    const h = img.offsetHeight
    if (w > 0 && h > 0 && imageSize.width > 0 && imageSize.height > 0) {
      setDisplaySize({ width: w, height: h })
      scaleRef.current = {
        scaleX: w / imageSize.width,
        scaleY: h / imageSize.height,
      }
    }
  }, [imageSize.width, imageSize.height])

  useEffect(() => {
    updateDisplaySize()
    const img = imgRef.current
    const ro = new ResizeObserver(updateDisplaySize)
    if (img) ro.observe(img)
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [updateDisplaySize])

  // To image coords from display
  const toImage = useCallback(
    (dx: number, dy: number): Point => ({
      x: dx / scaleRef.current.scaleX,
      y: dy / scaleRef.current.scaleY,
    }),
    []
  )
  const toDisplay = useCallback(
    (p: Point): { x: number; y: number } => ({
      x: p.x * scaleRef.current.scaleX,
      y: p.y * scaleRef.current.scaleY,
    }),
    []
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || displaySize.width === 0) return

    canvas.width = displaySize.width
    canvas.height = displaySize.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, displaySize.width, displaySize.height)

    if (points.length >= 2) {
      ctx.strokeStyle = 'rgba(63, 185, 80, 0.9)'
      ctx.lineWidth = 2
      ctx.beginPath()
      const first = toDisplay(points[0])
      ctx.moveTo(first.x, first.y)
      for (let i = 1; i < points.length; i++) {
        const p = toDisplay(points[i])
        ctx.lineTo(p.x, p.y)
      }
      ctx.closePath()
      ctx.stroke()
      ctx.fillStyle = 'rgba(63, 185, 80, 0.15)'
      ctx.fill()
    }

    // Padding outline (expanded contour)
    if (points.length >= MIN_POINTS && paddingPx > 0) {
      const pad = paddingPx * Math.min(scaleRef.current.scaleX, scaleRef.current.scaleY)
      const minX = Math.min(...points.map((p) => toDisplay(p).x)) - pad
      const minY = Math.min(...points.map((p) => toDisplay(p).y)) - pad
      const maxX = Math.max(...points.map((p) => toDisplay(p).x)) + pad
      const maxY = Math.max(...points.map((p) => toDisplay(p).y)) + pad
      ctx.strokeStyle = 'rgba(248, 81, 73, 0.5)'
      ctx.lineWidth = 1
      ctx.setLineDash([4, 4])
      ctx.strokeRect(minX, minY, maxX - minX, maxY - minY)
      ctx.setLineDash([])
    }

    points.forEach((p, i) => {
      const d = toDisplay(p)
      ctx.fillStyle = dragIndex === i ? '#f85149' : '#3fb950'
      ctx.beginPath()
      ctx.arc(d.x, d.y, HANDLE_R, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#0f1419'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }, [points, paddingPx, displaySize, toDisplay, dragIndex])

  const getEventPoint = (e: React.MouseEvent) => {
    const img = imgRef.current
    if (!img) return null
    const rect = img.getBoundingClientRect()
    const dx = e.clientX - rect.left
    const dy = e.clientY - rect.top
    if (dx < 0 || dy < 0 || dx > rect.width || dy > rect.height) return null
    const pt = toImage(dx, dy)
    // Не добавлять точки вне изображения
    if (pt.x < 0 || pt.y < 0 || pt.x >= imageSize.width || pt.y >= imageSize.height) return null
    return pt
  }

  const hitTest = (imgPoint: Point): number => {
    const s = Math.max(scaleRef.current.scaleX, scaleRef.current.scaleY)
    const r = (HANDLE_R + 2) / s
    for (let i = 0; i < points.length; i++) {
      const dx = points[i].x - imgPoint.x
      const dy = points[i].y - imgPoint.y
      if (dx * dx + dy * dy <= r * r) return i
    }
    return -1
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return
    const imgP = getEventPoint(e)
    if (!imgP) return
    const i = hitTest(imgP)
    if (i >= 0) {
      setDragIndex(i)
      return
    }
    if (e.button === 0) {
      onChange([...points, imgP])
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragIndex !== null && !disabled) {
      const imgP = getEventPoint(e)
      if (!imgP) return
      const next = [...points]
      next[dragIndex] = imgP
      onChange(next)
    }
  }

  const handleMouseUp = () => {
    setDragIndex(null)
  }

  const handleMouseLeave = () => {
    setDragIndex(null)
  }

  const clearContour = () => {
    onChange([])
  }

  const removeLastPoint = () => {
    if (points.length > 0) onChange(points.slice(0, -1))
  }

  return (
    <div className="contour-editor">
      <div
        ref={containerRef}
        className="contour-editor__wrap"
        style={{ aspectRatio: `${imageSize.width}/${imageSize.height}` }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Эталон"
          className="contour-editor__img"
          style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
          draggable={false}
          onLoad={updateDisplaySize}
        />
        <canvas
          ref={canvasRef}
          className="contour-editor__canvas"
          style={{ width: displaySize.width, height: displaySize.height }}
        />
      </div>
      <p className="contour-editor__hint">
        Клик — добавить точку контура. Перетаскивайте точки. Зелёная область — зона анализа, пунктир — с учётом запаса.
      </p>
      <div className="contour-editor__actions">
        <button type="button" onClick={removeLastPoint} disabled={points.length === 0 || disabled}>
          Удалить последнюю точку
        </button>
        <button type="button" onClick={clearContour} disabled={points.length === 0 || disabled}>
          Очистить контур
        </button>
      </div>
    </div>
  )
}
