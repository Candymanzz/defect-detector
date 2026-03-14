import type { ReactNode } from 'react'

interface AnalyzeButtonProps {
  onClick: () => void
  analyzing: boolean
  disabled: boolean
  disabledReason?: ReactNode
}

export function AnalyzeButton({
  onClick,
  analyzing,
  disabled,
  disabledReason,
}: AnalyzeButtonProps) {
  return (
    <div className="analyze-row">
      <button
        type="button"
        className="analyze-btn"
        onClick={onClick}
        disabled={disabled || analyzing}
      >
        {analyzing ? 'Анализирую…' : 'Анализировать'}
      </button>
      {disabled && disabledReason && (
        <p className="analyze-row__hint">{disabledReason}</p>
      )}
    </div>
  )
}
