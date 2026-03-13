import { useState, useCallback } from 'react'
import './App.css'
import { ReferenceStep } from './components/ReferenceStep'
import { TestStep } from './components/TestStep'
import { DefectHighlight } from './components/DefectHighlight'
import { compareWithReference } from './lib/compare'
import type { Point } from './lib/compare'
import type { CompareResult } from './lib/compare'

type Step = 'reference' | 'test'

function App() {
  const [step, setStep] = useState<Step>('reference')
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null)
  const [referenceImageSize, setReferenceImageSize] = useState({ width: 0, height: 0 })
  const [contourPoints, setContourPoints] = useState<Point[]>([])
  const [paddingPx, setPaddingPx] = useState(0)
  const [testImageUrl, setTestImageUrl] = useState<string | null>(null)
  const [testImageSize, setTestImageSize] = useState({ width: 0, height: 0 })
  const [defectThresholdMse, setDefectThresholdMse] = useState(120)
  const [ignoreDiffThreshold, setIgnoreDiffThreshold] = useState(6)
  const [comparing, setComparing] = useState(false)
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)

  // Anomalib (Intel) — опциональный бэкенд
  const [anomalibBackendUrl, setAnomalibBackendUrl] = useState('http://localhost:8000')
  const [anomalibThreshold, setAnomalibThreshold] = useState(0.5)
  const [anomalibComparing, setAnomalibComparing] = useState(false)
  const [anomalibResult, setAnomalibResult] = useState<{
    defect: boolean
    score: number
    threshold: number
    raw_score?: number | null
    heatmap_base64: string | null
    message?: string
  } | null>(null)

  // Совмещённый вывод: объединяем MSE-сравнение и Patchcore
  const combinedResult = (() => {
    if (!compareResult && !anomalibResult) return null
    const mseDefect = compareResult?.defect ?? null
    const patchcoreDefect =
      anomalibResult != null ? anomalibResult.score >= anomalibResult.threshold : null
    const finalDefect = (mseDefect ?? false) || (patchcoreDefect ?? false)
    let note = ''
    if (mseDefect != null && patchcoreDefect != null && mseDefect !== patchcoreDefect) {
      note =
        'Классические сравнение и Patchcore расходятся: проверьте зону дефекта по теплокарте и по контуру.'
    }
    return { finalDefect, mseDefect, patchcoreDefect, note }
  })()

  const handleReferenceSelected = useCallback((url: string, width: number, height: number) => {
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl)
    setReferenceImageUrl(url)
    setReferenceImageSize({ width, height })
    setContourPoints([])
    setCompareResult(null)
    setAnomalibResult(null)
  }, [referenceImageUrl])

  const handleTestImageLoad = useCallback((url: string, width?: number, height?: number) => {
    if (testImageUrl) URL.revokeObjectURL(testImageUrl)
    setTestImageUrl(url)
    setTestImageSize({ width: width ?? 0, height: height ?? 0 })
    setCompareResult(null)
    setAnomalibResult(null)
  }, [testImageUrl])

  const handleCompare = useCallback(() => {
    if (!referenceImageUrl || !testImageUrl || contourPoints.length < 3) return
    setComparing(true)
    setCompareResult(null)

    const refImg = new Image()
    const testImg = new Image()

    const run = () => {
      const result = compareWithReference(
        refImg,
        testImg,
        contourPoints,
        paddingPx,
        defectThresholdMse,
        ignoreDiffThreshold
      )
      setCompareResult(result)
      setComparing(false)
    }

    testImg.onload = run
    refImg.onload = () => {
      testImg.src = testImageUrl
    }
    refImg.onerror = () => {
      setComparing(false)
    }
    testImg.onerror = () => {
      setComparing(false)
    }
    refImg.src = referenceImageUrl
  }, [
    referenceImageUrl,
    testImageUrl,
    contourPoints,
    paddingPx,
    defectThresholdMse,
    ignoreDiffThreshold,
  ])

  const handleCompareAnomalib = useCallback(async () => {
    if (!referenceImageUrl || !testImageUrl) return
    setAnomalibComparing(true)
    setAnomalibResult(null)
    try {
      const [refRes, testRes] = await Promise.all([
        fetch(referenceImageUrl),
        fetch(testImageUrl),
      ])
      if (!refRes.ok || !testRes.ok) throw new Error('Не удалось загрузить изображения')
      const refBlob = await refRes.blob()
      const testBlob = await testRes.blob()
      const base = anomalibBackendUrl.replace(/\/$/, '')
      const form = new FormData()
      form.append('reference', refBlob, 'reference.png')
      form.append('test', testBlob, 'test.png')
      form.append('threshold', String(anomalibThreshold))
      const r = await fetch(`${base}/api/anomalib/analyze`, {
        method: 'POST',
        body: form,
      })
      if (!r.ok) {
        const t = await r.text()
        let msg = `Ошибка ${r.status}`
        try {
          const errJson = JSON.parse(t)
          if (typeof errJson.detail === 'string') msg = errJson.detail
        } catch {
          if (t) msg = t
        }
        throw new Error(msg)
      }
      const data = await r.json()
      setAnomalibResult({
        defect: data.defect,
        score: data.score,
        threshold: data.threshold,
        raw_score: data.raw_score ?? null,
        heatmap_base64: data.heatmap_base64 ?? null,
        message: data.message,
      })
    } catch (e) {
      setAnomalibResult({
        defect: true,
        score: 1,
        threshold: anomalibThreshold,
        heatmap_base64: null,
        message: e instanceof Error ? e.message : 'Ошибка запроса к бэкенду Anomalib',
      })
    } finally {
      setAnomalibComparing(false)
    }
  }, [referenceImageUrl, testImageUrl, anomalibBackendUrl, anomalibThreshold])

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Детектор дефектов этикетки на ведре</h1>
        <p className="app__subtitle">
          Эталон → контур зоны анализа → тестовое фото → сравнение с эталоном
        </p>
      </header>

      <main className="app__main">
        <ReferenceStep
          referenceImageUrl={referenceImageUrl}
          referenceImageSize={referenceImageSize}
          onReferenceSelected={handleReferenceSelected}
          contourPoints={contourPoints}
          onContourChange={setContourPoints}
          paddingPx={paddingPx}
          onPaddingChange={setPaddingPx}
          onNext={() => setStep('test')}
        />

        {step === 'test' && (
          <>
            <TestStep
              testImageUrl={testImageUrl}
              onTestImageLoad={handleTestImageLoad}
              onCompare={handleCompare}
              comparing={comparing}
            />

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
                onChange={(e) => setDefectThresholdMse(Number(e.target.value))}
              />
              <span className="step__hint">Выше — меньше ложных «браков», ниже — чувствительнее к отличиям. Для чёрного ведра с белой этикеткой лучше 150–300.</span>

              <label className="step__threshold-row">
                Игнорировать мелкие отличия: <strong>{ignoreDiffThreshold}</strong> (уровней яркости)
              </label>
              <input
                type="range"
                min={0}
                max={25}
                step={1}
                value={ignoreDiffThreshold}
                onChange={(e) => setIgnoreDiffThreshold(Number(e.target.value))}
              />
              <span className="step__hint">Разница по яркости меньше этого не считается. Для белой этикетки ставьте 5–8, чтобы ловить тёмные пятна и царапины.</span>
            </div>

            <div className="step step--anomalib">
              <h3 className="step__subtitle">Режим Anomalib (Patchcore)</h3>
              <p className="step__desc">
                Сравнение через нейросеть Patchcore: модель обучена на локальном датасете server/dataset/normal и ищет локальные дефекты (царапины) на тестовом фото. Запустите сервер из папки <code>server</code>.
              </p>
              <label className="step__threshold-row">
                URL бэкенда:{' '}
                <input
                  type="url"
                  value={anomalibBackendUrl}
                  onChange={(e) => setAnomalibBackendUrl(e.target.value)}
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
                  value={anomalibThreshold}
                  onChange={(e) => {
                  const v = Number(e.target.value);
                  setAnomalibThreshold(Number.isNaN(v) ? 0 : Math.max(0, Math.min(1, v)));
                }}
                  className="step__input-number"
                />
              </label>
              <button
                type="button"
                className="step__compare step__compare--anomalib"
                onClick={handleCompareAnomalib}
                disabled={!testImageUrl || anomalibComparing}
              >
                {anomalibComparing ? 'Anomalib анализирует…' : 'Сравнить через Anomalib'}
              </button>
            </div>

            {anomalibResult && (
              <div className="result-block result-block--anomalib">
                <div className={`result ${anomalibResult.defect ? 'result--defect' : 'result--ok'}`}>
                  <span className="result__label">Anomalib:</span>
                  <span className="result__value">
                    {anomalibResult.defect ? 'Брак' : 'Нет брака'}
                  </span>
                </div>
                <p className="result-block__mse">
                  score (нормализован 0–1) = {anomalibResult.score.toFixed(4)}
                  {anomalibResult.raw_score != null && (
                    <> · сырой Padim = {Number(anomalibResult.raw_score).toFixed(2)}</>
                  )}
                  {' '}(порог {anomalibResult.threshold} → {anomalibResult.score >= anomalibResult.threshold ? 'брак' : 'норма'})
                </p>
                {anomalibResult.message && (
                  <p className="result-block__message">{anomalibResult.message}</p>
                )}
                {anomalibResult.heatmap_base64 && (
                  <div className="result-block__heatmap">
                    <p className="defect-highlight__caption">Карта аномалий (Patchcore)</p>
                    <img
                      src={`data:image/png;base64,${anomalibResult.heatmap_base64}`}
                      alt="Heatmap"
                      className="result-block__heatmap-img"
                    />
                  </div>
                )}
              </div>
            )}

            {compareResult && (
              <div className="result-block">
                <div className={`result ${compareResult.defect ? 'result--defect' : 'result--ok'}`}>
                  <span className="result__label">Результат:</span>
                  <span className="result__value">
                    {compareResult.defect ? 'Брак' : 'Нет брака'}
                  </span>
                </div>
                <p className="result-block__mse">
                  MSE = {compareResult.mse.toFixed(1)} (порог {compareResult.threshold})
                </p>
                {testImageUrl && compareResult.matchPosition && (
                  <DefectHighlight
                    testImageUrl={testImageUrl}
                    testImageSize={testImageSize}
                    compareResult={compareResult}
                  />
                )}
              </div>
            )}

            {combinedResult && (
              <div className="result-block result-block--combined">
                <div
                  className={`result ${
                    combinedResult.finalDefect ? 'result--defect' : 'result--ok'
                  }`}
                >
                  <span className="result__label">Совмещённый вывод:</span>
                  <span className="result__value">
                    {combinedResult.finalDefect ? 'Брак' : 'Нет брака'}
                  </span>
                </div>
                <p className="result-block__mse">
                  MSE → {combinedResult.mseDefect == null ? 'нет данных' : combinedResult.mseDefect ? 'брак' : 'норма'}
                  {' · '}Patchcore →
                  {combinedResult.patchcoreDefect == null
                    ? 'нет данных'
                    : combinedResult.patchcoreDefect
                    ? 'брак'
                    : 'норма'}
                </p>
                {combinedResult.note && (
                  <p className="result-block__message">{combinedResult.note}</p>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
