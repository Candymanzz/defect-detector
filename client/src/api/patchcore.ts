/**
 * API бэкенда Patchcore (Anomalib). Анализ изображения на дефекты.
 */

export interface PatchcoreAnalyzeResult {
  defect: boolean
  score: number
  threshold: number
  raw_score?: number | null
  heatmap_base64: string | null
  message?: string
}

export interface PatchcoreAnalyzeError {
  defect: true
  score: number
  threshold: number
  heatmap_base64: null
  message: string
}

/**
 * Отправляет эталон и тест на анализ, возвращает результат или ошибку в том же формате.
 */
export async function analyzePatchcore(
  apiBaseUrl: string,
  referenceBlob: Blob,
  testBlob: Blob,
  threshold: number
): Promise<PatchcoreAnalyzeResult | PatchcoreAnalyzeError> {
  const form = new FormData()
  form.append('reference', referenceBlob, 'reference.png')
  form.append('test', testBlob, 'test.png')
  form.append('threshold', String(threshold))

  const r = await fetch(`${apiBaseUrl}/api/anomalib/analyze`, {
    method: 'POST',
    body: form,
  })

  if (!r.ok) {
    const text = await r.text()
    let msg = `Ошибка ${r.status}`
    try {
      const err = JSON.parse(text)
      if (typeof err.detail === 'string') msg = err.detail
    } catch {
      if (text) msg = text
    }
    return {
      defect: true,
      score: 1,
      threshold,
      heatmap_base64: null,
      message: msg,
    }
  }

  const data = await r.json()
  return {
    defect: data.defect,
    score: data.score,
    threshold: data.threshold,
    raw_score: data.raw_score ?? null,
    heatmap_base64: data.heatmap_base64 ?? null,
    message: data.message,
  }
}
