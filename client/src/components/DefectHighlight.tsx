import { useRef, useEffect, useState } from 'react'
import type { CompareResult } from '../lib/compare'

interface DefectHighlightProps {
  testImageUrl: string
  testImageSize: { width: number; height: number }
  compareResult: CompareResult
}

export function DefectHighlight({
  testImageUrl,
  testImageSize,
  compareResult,
}: DefectHighlightProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const img = imgRef.current
    if (!img || !compareResult.matchPosition) return

    const updateSize = () => {
      if (img.clientWidth > 0 && img.clientHeight > 0) {
        setDisplaySize({ width: img.clientWidth, height: img.clientHeight })
      }
    }

    updateSize()
    if (!img.complete) img.addEventListener('load', updateSize)
    const t = setTimeout(updateSize, 150)
    return () => {
      if (!img.complete) img.removeEventListener('load', updateSize)
      clearTimeout(t)
    }
  }, [testImageUrl, compareResult.matchPosition, compareResult])

  useEffect(() => {
    const canvas = canvasRef.current
    const { matchPosition, roiSize, defect, diffMap, smoothMap, contourMask, ignoreDiffUsed } = compareResult
    if (!canvas || displaySize.width === 0 || !matchPosition || roiSize.width === 0) return

    const scaleX = displaySize.width / testImageSize.width
    const scaleY = displaySize.height / testImageSize.height

    canvas.width = displaySize.width
    canvas.height = displaySize.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const x = matchPosition.x * scaleX
    const y = matchPosition.y * scaleY
    const w = roiSize.width * scaleX
    const h = roiSize.height * scaleY

    ctx.clearRect(0, 0, displaySize.width, displaySize.height)

    const roiW = roiSize.width
    const roiH = roiSize.height
    const expectedLen = roiW * roiH
    const hasDiffMap =
      defect &&
      diffMap &&
      typeof diffMap.length === 'number' &&
      diffMap.length >= expectedLen
    const hasSmoothMap =
      smoothMap &&
      typeof smoothMap.length === 'number' &&
      smoothMap.length >= expectedLen
    const hasContourMask =
      contourMask &&
      typeof contourMask.length === 'number' &&
      contourMask.length >= expectedLen

    if (hasDiffMap) {
      const DIFF_THRESHOLD = 38
      const SMOOTH_THRESHOLD = 55
      const thresholdVis = Math.max(ignoreDiffUsed ?? 0, DIFF_THRESHOLD)
      const minSpot = 5

      const candidates: { idx: number; diff: number }[] = []
      for (let py = 0; py < roiH; py++) {
        for (let px = 0; px < roiW; px++) {
          const idx = py * roiW + px
          if (idx >= diffMap.length) continue
          if (hasContourMask && contourMask[idx] === 0) continue
          const diffOk = diffMap[idx] >= thresholdVis
          const smoothOk = !hasSmoothMap || (smoothMap[idx] <= SMOOTH_THRESHOLD)
          if (diffOk && smoothOk) candidates.push({ idx, diff: diffMap[idx] })
        }
      }
      const topFraction = 0.06
      const topCount = Math.max(30, Math.ceil(candidates.length * topFraction))
      candidates.sort((a, b) => b.diff - a.diff)
      const toDraw = new Set(candidates.slice(0, topCount).map((c) => c.idx))

      const pxW = Math.max(minSpot, w / roiW)
      const pxH = Math.max(minSpot, h / roiH)
      for (let py = 0; py < roiH; py++) {
        for (let px = 0; px < roiW; px++) {
          const idx = py * roiW + px
          if (hasContourMask && contourMask[idx] === 0) continue
          if (!toDraw.has(idx)) continue
          const dx = x + px * (w / roiW)
          const dy = y + py * (h / roiH)
          ctx.fillStyle = 'rgba(255, 70, 70, 0.9)'
          ctx.fillRect(dx, dy, pxW, pxH)
        }
      }
    } else if (defect) {
      // Нет карты отличий — рисуем контур или рамку зоны
      if (hasContourMask) {
        ctx.fillStyle = 'rgba(248, 81, 73, 0.4)'
        for (let py = 0; py < roiH; py++) {
          for (let px = 0; px < roiW; px++) {
            if (contourMask[py * roiW + px] === 0) continue
            const dx = x + px * (w / roiW)
            const dy = y + py * (h / roiH)
            ctx.fillRect(dx, dy, Math.max(1, w / roiW), Math.max(1, h / roiH))
          }
        }
      } else {
        ctx.strokeStyle = 'rgba(248, 81, 73, 0.9)'
        ctx.lineWidth = 3
        ctx.strokeRect(x, y, w, h)
      }
    } else {
      if (hasContourMask) {
        ctx.fillStyle = 'rgba(63, 185, 80, 0.35)'
        for (let py = 0; py < roiH; py++) {
          for (let px = 0; px < roiW; px++) {
            if (contourMask[py * roiW + px] === 0) continue
            const dx = x + px * (w / roiW)
            const dy = y + py * (h / roiH)
            ctx.fillRect(dx, dy, Math.max(1, w / roiW), Math.max(1, h / roiH))
          }
        }
      } else {
        ctx.strokeStyle = 'rgba(63, 185, 80, 0.95)'
        ctx.lineWidth = 4
        ctx.strokeRect(x, y, w, h)
        ctx.fillStyle = 'rgba(63, 185, 80, 0.15)'
        ctx.fillRect(x, y, w, h)
      }
    }
  }, [compareResult, displaySize, testImageSize, testImageUrl])

  if (!compareResult.matchPosition || compareResult.roiSize.width === 0) return null

  const roiW = compareResult.roiSize.width
  const roiH = compareResult.roiSize.height
  const expectedLen = roiW * roiH
  const hasDiffMap =
    compareResult.defect &&
    compareResult.diffMap &&
    typeof compareResult.diffMap.length === 'number' &&
    compareResult.diffMap.length >= expectedLen

  return (
    <div className="defect-highlight">
      <p className="defect-highlight__caption">
        {compareResult.defect
            ? hasDiffMap
            ? 'Красным отмечены самые сильные отличия (топ по контрасту)'
            : 'Брак есть, но карта отличий не загрузилась. Обновите страницу (Ctrl+Shift+R) и нажмите «Сравнить с эталоном» ещё раз.'
          : 'Область проверки: без брака'}
      </p>
      <div ref={wrapRef} className="defect-highlight__wrap">
        <img
          ref={imgRef}
          src={testImageUrl}
          alt="Тестовое фото с выделенной областью"
          className="defect-highlight__img"
        />
        <canvas
          ref={canvasRef}
          className="defect-highlight__canvas"
          style={{ width: displaySize.width, height: displaySize.height }}
        />
      </div>
    </div>
  )
}
