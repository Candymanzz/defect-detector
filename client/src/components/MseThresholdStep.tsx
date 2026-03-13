import type { ChangeEvent } from 'react'

interface MseThresholdStepProps {
  defectThresholdMse: number
  onDefectThresholdMseChange: (v: number) => void
  ignoreDiffThreshold: number
  onIgnoreDiffThresholdChange: (v: number) => void
}

export function MseThresholdStep({
  defectThresholdMse,
  onDefectThresholdMseChange,
  ignoreDiffThreshold,
  onIgnoreDiffThresholdChange,
}: MseThresholdStepProps) {
  const handleMse = (e: ChangeEvent<HTMLInputElement>) =>
    onDefectThresholdMseChange(Number(e.target.value))
  const handleIgnore = (e: ChangeEvent<HTMLInputElement>) =>
    onIgnoreDiffThresholdChange(Number(e.target.value))

  return (
    <div className="step step--threshold">
      <label>
        Порог чувствительности (MSE): <strong>{defectThresholdMse}</strong>
      </label>
      <input
        type="range"
        min={50}
        max={2000}
        step={50}
        value={defectThresholdMse}
        onChange={handleMse}
      />
      <span className="step__hint">
        Выше — меньше ложных «браков», ниже — чувствительнее к отличиям. Для чёрного ведра с белой
        этикеткой лучше 150–300.
      </span>

      <label className="step__threshold-row">
        Игнорировать мелкие отличия: <strong>{ignoreDiffThreshold}</strong> (уровней яркости)
      </label>
      <input
        type="range"
        min={0}
        max={25}
        step={1}
        value={ignoreDiffThreshold}
        onChange={handleIgnore}
      />
      <span className="step__hint">
        Разница по яркости меньше этого не считается. Для белой этикетки ставьте 5–8, чтобы ловить
        тёмные пятна и царапины.
      </span>
    </div>
  )
}
