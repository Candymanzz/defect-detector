import { useState, useCallback } from 'react'

interface TestStepProps {
  testImageUrl: string | null
  onTestImageLoad: (url: string, width: number, height: number) => void
  onCompare: () => void
  comparing: boolean
}

export function TestStep({
  testImageUrl,
  onTestImageLoad,
  onCompare,
  comparing,
}: TestStepProps) {
  const [fileError, setFileError] = useState<string | null>(null)

  const handleFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFileError(null)
      const file = e.target.files?.[0]
      if (!file) return
      const url = URL.createObjectURL(file)
      const img = new Image()
      img.onload = () => {
        onTestImageLoad(url, img.naturalWidth, img.naturalHeight)
      }
      img.onerror = () => {
        setFileError('Не удалось загрузить изображение')
      }
      img.src = url
      e.target.value = ''
    },
    [onTestImageLoad]
  )

  return (
    <section className="step">
      <h2 className="step__title">2. Тестовое фото</h2>
      <p className="step__desc">Загрузите одно фото ведра для проверки (та же сторона с этикеткой).</p>

      <label className="step__upload">
        <span>Выбрать тестовое фото</span>
        <input type="file" accept="image/*" onChange={handleFile} />
      </label>
      {fileError && <p className="step__error">{fileError}</p>}

      {testImageUrl && (
        <div className="step__preview">
          <img src={testImageUrl} alt="Тест" className="step__preview-img" />
        </div>
      )}

      <button
        type="button"
        className="step__compare"
        onClick={onCompare}
        disabled={!testImageUrl || comparing}
      >
        {comparing ? 'Сравниваю…' : 'Сравнить с эталоном'}
      </button>
    </section>
  )
}
