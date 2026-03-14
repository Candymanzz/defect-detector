import { useState, useCallback } from 'react'
import { ContourEditor } from './ContourEditor'
import type { Point } from '../lib/compare'

interface ReferenceBlockProps {
  referenceImageUrl: string | null
  referenceImageSize: { width: number; height: number }
  onReferenceSelected: (url: string, width: number, height: number) => void
  contourPoints: Point[]
  onContourChange: (points: Point[]) => void
  paddingPx: number
  onPaddingChange: (v: number) => void
}

export function ReferenceBlock({
  referenceImageUrl,
  referenceImageSize,
  onReferenceSelected,
  contourPoints,
  onContourChange,
  paddingPx,
  onPaddingChange,
}: ReferenceBlockProps) {
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

  const hasContour = contourPoints.length >= 3
  const canUseMse = referenceImageUrl != null && hasContour

  return (
    <section className="block reference-block">
      <h2 className="block__title">Эталон (опционально)</h2>
      <p className="block__desc">
        Нужен только для сравнения по алгоритму MSE. Для анализа нейросетью достаточно тестового
        изображения.
      </p>

      {!referenceImageUrl ? (
        <>
          <label className="block__upload">
            <span>Выбрать эталонное фото</span>
            <input type="file" accept="image/*" onChange={handleFile} />
          </label>
          <div className="reference-block__badge reference-block__badge--missing">
            Эталон не выбран — сравнение по алгоритму недоступно
          </div>
        </>
      ) : (
        <>
          <div className="reference-block__badge reference-block__badge--ok">
            Эталон загружен {hasContour ? '· контур задан' : '· задайте контур (≥3 точки)'}
          </div>
          {!hasContour && (
            <p className="reference-block__warn">Сравнение по алгоритму будет недоступно до задания контура.</p>
          )}
          <label className="block__upload block__upload--secondary">
            <span>Заменить эталон</span>
            <input type="file" accept="image/*" onChange={handleFile} />
          </label>
          <div className="block__contour">
            <ContourEditor
              imageUrl={referenceImageUrl}
              imageSize={referenceImageSize}
              points={contourPoints}
              onChange={onContourChange}
              paddingPx={paddingPx}
            />
          </div>
          <div className="block__padding">
            <label>
              Запас по контуру (px): <strong>{paddingPx}</strong>
            </label>
            <input
              type="range"
              min={0}
              max={80}
              value={paddingPx}
              onChange={(e) => onPaddingChange(Number(e.target.value))}
            />
          </div>
          {fileError && <p className="block__error">{fileError}</p>}
        </>
      )}
    </section>
  )
}
