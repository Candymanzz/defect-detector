import { useState, useCallback } from 'react'
import { ContourEditor } from './ContourEditor'
import type { Point } from '../lib/compare'

interface ReferenceStepProps {
  referenceImageUrl: string | null
  referenceImageSize: { width: number; height: number }
  onReferenceSelected: (url: string, width: number, height: number) => void
  contourPoints: Point[]
  onContourChange: (points: Point[]) => void
  paddingPx: number
  onPaddingChange: (v: number) => void
  onNext: () => void
}

export function ReferenceStep({
  referenceImageUrl,
  referenceImageSize,
  onReferenceSelected,
  contourPoints,
  onContourChange,
  paddingPx,
  onPaddingChange,
  onNext,
}: ReferenceStepProps) {
  const [fileError, setFileError] = useState<string | null>(null)

  const handleFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFileError(null)
      const file = e.target.files?.[0]
      if (!file) return
      const url = URL.createObjectURL(file)
      const img = new Image()
      img.onload = () => {
        onReferenceSelected(url, img.naturalWidth, img.naturalHeight)
      }
      img.onerror = () => {
        setFileError('Не удалось загрузить изображение')
        URL.revokeObjectURL(url)
      }
      img.src = url
      e.target.value = ''
    },
    [onReferenceSelected]
  )

  const canNext = referenceImageUrl && contourPoints.length >= 3

  return (
    <section className="step">
      <h2 className="step__title">1. Эталон «хорошего» ведра</h2>
      <p className="step__desc">Загрузите фото ведра без дефектов (одна сторона с этикеткой).</p>

      <label className="step__upload">
        <span>Выбрать эталонное фото</span>
        <input type="file" accept="image/*" onChange={handleFile} />
      </label>
      {fileError && <p className="step__error">{fileError}</p>}

      {referenceImageUrl && (
        <>
          <div className="step__contour">
            <ContourEditor
              imageUrl={referenceImageUrl}
              imageSize={referenceImageSize}
              points={contourPoints}
              onChange={onContourChange}
              paddingPx={paddingPx}
            />
          </div>
          <div className="step__padding">
            <label>
              Запас по контуру (пиксели): <strong>{paddingPx}</strong>
            </label>
            <input
              type="range"
              min={0}
              max={80}
              value={paddingPx}
              onChange={(e) => onPaddingChange(Number(e.target.value))}
            />
            <span className="step__hint">В пикселях изображения. Зона анализа = прямоугольник по границам контура + запас. При 0 анализируется только этот прямоугольник.</span>
          </div>
          <button type="button" className="step__next" onClick={onNext} disabled={!canNext}>
            Контур задан — перейти к тесту
          </button>
        </>
      )}
    </section>
  )
}
