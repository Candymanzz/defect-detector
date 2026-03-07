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

  const handleReferenceSelected = useCallback((url: string, width: number, height: number) => {
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl)
    setReferenceImageUrl(url)
    setReferenceImageSize({ width, height })
    setContourPoints([])
    setCompareResult(null)
  }, [referenceImageUrl])

  const handleTestImageLoad = useCallback((url: string, width?: number, height?: number) => {
    if (testImageUrl) URL.revokeObjectURL(testImageUrl)
    setTestImageUrl(url)
    setTestImageSize({ width: width ?? 0, height: height ?? 0 })
    setCompareResult(null)
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
          </>
        )}
      </main>
    </div>
  )
}

export default App
