import { useState, useCallback } from 'react'

interface TestBlockProps {
  testImageUrl: string | null
  onTestImageLoad: (url: string, width: number, height: number) => void
}

export function TestBlock({ testImageUrl, onTestImageLoad }: TestBlockProps) {
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
    <section className="block test-block">
      <h2 className="block__title">Тестовое изображение</h2>
      <p className="block__desc">Обязательно для анализа. Используется и нейросетью, и алгоритмом (если выбран эталон).</p>

      <label className="block__upload">
        <span>{testImageUrl ? 'Заменить тестовое фото' : 'Выбрать тестовое фото'}</span>
        <input type="file" accept="image/*" onChange={handleFile} />
      </label>
      {fileError && <p className="block__error">{fileError}</p>}
      {testImageUrl && (
        <div className="test-block__preview">
          <img src={testImageUrl} alt="Тест" />
        </div>
      )}
    </section>
  )
}
