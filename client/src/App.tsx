import { useState, useCallback } from 'react'
import { config } from './config'
import { analyzePatchcore } from './api/patchcore'
import type { PatchcoreAnalyzeResult } from './api/patchcore'
import type { Point, CompareResult } from './types'
import { compareWithReference } from './lib/compare'
import { ReferenceStep } from './components/ReferenceStep'
import { TestStep } from './components/TestStep'
import { MseThresholdStep } from './components/MseThresholdStep'
import { PatchcoreStep } from './components/PatchcoreStep'
import { ResultBlocks } from './components/ResultBlocks'
import './App.css'

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

  const [apiBaseUrl, setApiBaseUrl] = useState(config.apiBaseUrl)
  const [patchcoreThreshold, setPatchcoreThreshold] = useState(0.5)
  const [patchcoreAnalyzing, setPatchcoreAnalyzing] = useState(false)
  const [patchcoreResult, setPatchcoreResult] = useState<PatchcoreAnalyzeResult | null>(null)

  const handleReferenceSelected = useCallback((url: string, width: number, height: number) => {
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl)
    setReferenceImageUrl(url)
    setReferenceImageSize({ width, height })
    setContourPoints([])
    setCompareResult(null)
    setPatchcoreResult(null)
  }, [referenceImageUrl])

  const handleTestImageLoad = useCallback((url: string, width?: number, height?: number) => {
    if (testImageUrl) URL.revokeObjectURL(testImageUrl)
    setTestImageUrl(url)
    setTestImageSize({ width: width ?? 0, height: height ?? 0 })
    setCompareResult(null)
    setPatchcoreResult(null)
  }, [testImageUrl])

  const handleCompare = useCallback(() => {
    if (!referenceImageUrl || !testImageUrl || contourPoints.length < 3) return
    setComparing(true)
    setCompareResult(null)
    const refImg = new Image()
    const testImg = new Image()
    const run = () => {
      setCompareResult(
        compareWithReference(
          refImg,
          testImg,
          contourPoints,
          paddingPx,
          defectThresholdMse,
          ignoreDiffThreshold
        )
      )
      setComparing(false)
    }
    testImg.onload = run
    refImg.onload = () => { testImg.src = testImageUrl }
    refImg.onerror = () => setComparing(false)
    testImg.onerror = () => setComparing(false)
    refImg.src = referenceImageUrl
  }, [
    referenceImageUrl,
    testImageUrl,
    contourPoints,
    paddingPx,
    defectThresholdMse,
    ignoreDiffThreshold,
  ])

  const handlePatchcoreAnalyze = useCallback(async () => {
    if (!referenceImageUrl || !testImageUrl) return
    setPatchcoreAnalyzing(true)
    setPatchcoreResult(null)
    try {
      const [refRes, testRes] = await Promise.all([
        fetch(referenceImageUrl),
        fetch(testImageUrl),
      ])
      if (!refRes.ok || !testRes.ok) throw new Error('Не удалось загрузить изображения')
      const refBlob = await refRes.blob()
      const testBlob = await testRes.blob()
      const result = await analyzePatchcore(
        apiBaseUrl,
        refBlob,
        testBlob,
        patchcoreThreshold
      )
      setPatchcoreResult(result)
    } finally {
      setPatchcoreAnalyzing(false)
    }
  }, [referenceImageUrl, testImageUrl, apiBaseUrl, patchcoreThreshold])

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

            <MseThresholdStep
              defectThresholdMse={defectThresholdMse}
              onDefectThresholdMseChange={setDefectThresholdMse}
              ignoreDiffThreshold={ignoreDiffThreshold}
              onIgnoreDiffThresholdChange={setIgnoreDiffThreshold}
            />

            <PatchcoreStep
              apiBaseUrl={apiBaseUrl}
              onApiBaseUrlChange={setApiBaseUrl}
              threshold={patchcoreThreshold}
              onThresholdChange={setPatchcoreThreshold}
              onAnalyze={handlePatchcoreAnalyze}
              analyzing={patchcoreAnalyzing}
              testImageUrl={testImageUrl}
            />

            <ResultBlocks
              testImageUrl={testImageUrl}
              testImageSize={testImageSize}
              compareResult={compareResult}
              anomalibResult={patchcoreResult}
            />
          </>
        )}
      </main>
    </div>
  )
}

export default App
