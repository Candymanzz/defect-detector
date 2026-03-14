import type { ChangeEvent } from 'react'

interface SidebarPanelProps {
  defectThresholdMse: number
  onDefectThresholdMseChange: (v: number) => void
  ignoreDiffThreshold: number
  onIgnoreDiffThresholdChange: (v: number) => void
  apiBaseUrl: string
  onApiBaseUrlChange: (v: string) => void
  patchcoreThreshold: number
  onPatchcoreThresholdChange: (v: number) => void
}

export function SidebarPanel({
  defectThresholdMse,
  onDefectThresholdMseChange,
  ignoreDiffThreshold,
  onIgnoreDiffThresholdChange,
  apiBaseUrl,
  onApiBaseUrlChange,
  patchcoreThreshold,
  onPatchcoreThresholdChange,
}: SidebarPanelProps) {
  const handleMse = (e: ChangeEvent<HTMLInputElement>) =>
    onDefectThresholdMseChange(Number(e.target.value))
  const handleIgnore = (e: ChangeEvent<HTMLInputElement>) =>
    onIgnoreDiffThresholdChange(Number(e.target.value))
  const handlePatch = (e: ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value)
    onPatchcoreThresholdChange(Number.isNaN(v) ? 0 : Math.max(0, Math.min(1, v)))
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__inner">
        <h3 className="sidebar__title">Настройки</h3>

        <section className="sidebar__section">
          <h4 className="sidebar__section-title">Алгоритм (MSE)</h4>
          <label className="sidebar__label">
            Порог MSE: <strong>{defectThresholdMse}</strong>
          </label>
          <input
            type="range"
            min={50}
            max={2000}
            step={50}
            value={defectThresholdMse}
            onChange={handleMse}
            className="sidebar__range"
          />
          <label className="sidebar__label">
            Игнор отличий: <strong>{ignoreDiffThreshold}</strong>
          </label>
          <input
            type="range"
            min={0}
            max={25}
            step={1}
            value={ignoreDiffThreshold}
            onChange={handleIgnore}
            className="sidebar__range"
          />
          <p className="sidebar__hint">Только при выбранном эталоне и контуре.</p>
        </section>

        <section className="sidebar__section">
          <h4 className="sidebar__section-title">Нейросеть (Patchcore)</h4>
          <label className="sidebar__label">URL бэкенда</label>
          <input
            type="url"
            value={apiBaseUrl}
            onChange={(e) => onApiBaseUrlChange(e.target.value)}
            placeholder="http://localhost:8000"
            className="sidebar__input"
          />
          <label className="sidebar__label">
            Порог score: <strong>{patchcoreThreshold}</strong>
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={patchcoreThreshold}
            onChange={handlePatch}
            className="sidebar__range"
          />
        </section>
      </div>
    </aside>
  )
}
