import type { ChangeEvent } from 'react'

interface PatchcoreStepProps {
  apiBaseUrl: string
  onApiBaseUrlChange: (v: string) => void
  threshold: number
  onThresholdChange: (v: number) => void
  onAnalyze: () => void
  analyzing: boolean
  testImageUrl: string | null
}

export function PatchcoreStep({
  apiBaseUrl,
  onApiBaseUrlChange,
  threshold,
  onThresholdChange,
  onAnalyze,
  analyzing,
  testImageUrl,
}: PatchcoreStepProps) {
  const handleThreshold = (e: ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value)
    onThresholdChange(Number.isNaN(v) ? 0 : Math.max(0, Math.min(1, v)))
  }

  return (
    <div className="step step--anomalib">
      <h3 className="step__subtitle">Режим Anomalib (Patchcore)</h3>
      <p className="step__desc">
        Сравнение через нейросеть Patchcore: модель обучена на датасете normal и ищет локальные
        дефекты на тестовом фото. Запустите бэкенд (см. README в корне репозитория).
      </p>
      <label className="step__threshold-row">
        URL бэкенда:{' '}
        <input
          type="url"
          value={apiBaseUrl}
          onChange={(e) => onApiBaseUrlChange(e.target.value)}
          placeholder="http://localhost:8000"
          className="step__input-url"
        />
      </label>
      <label className="step__threshold-row">
        Порог score (0–1):{' '}
        <input
          type="number"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={handleThreshold}
          className="step__input-number"
        />
      </label>
      <button
        type="button"
        className="step__compare step__compare--anomalib"
        onClick={onAnalyze}
        disabled={!testImageUrl || analyzing}
      >
        {analyzing ? 'Anomalib анализирует…' : 'Сравнить через Anomalib'}
      </button>
    </div>
  )
}
