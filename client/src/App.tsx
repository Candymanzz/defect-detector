import { useState, useCallback } from 'react'
import { config } from './config'
import { analyzePatchcore } from './api/patchcore'
import type { PatchcoreAnalyzeResult } from './api/patchcore'
import type { Point, CompareResult } from './types'
import { compareWithReference } from './lib/compare'
import { ReferenceBlock } from './components/ReferenceBlock'
import { TestBlock } from './components/TestBlock'
import { SidebarPanel } from './components/SidebarPanel'
import { AnalyzeButton } from './components/AnalyzeButton'
import { ResultBlocks } from './components/ResultBlocks'
import './App.css'

function App() {
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null)
  const [referenceImageSize, setReferenceImageSize] = useState({ width: 0, height: 0 })
  const [contourPoints, setContourPoints] = useState<Point[]>([])
  const [paddingPx, setPaddingPx] = useState(0)
  const [testImageUrl, setTestImageUrl] = useState<string | null>(null)
  const [testImageSize, setTestImageSize] = useState({ width: 0, height: 0 })

  const [defectThresholdMse, setDefectThresholdMse] = useState(120)
  const [ignoreDiffThreshold, setIgnoreDiffThreshold] = useState(6)
  const [apiBaseUrl, setApiBaseUrl] = useState(config.apiBaseUrl)
  const [patchcoreThreshold, setPatchcoreThreshold] = useState(0.62)

  const [mseRunning, setMseRunning] = useState(false)
  const [patchcoreRunning, setPatchcoreRunning] = useState(false)
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)
  const [patchcoreResult, setPatchcoreResult] = useState<PatchcoreAnalyzeResult | null>(null)
  /** Сообщение пользователю: почему не выполнен анализ по алгоритму */
  const [mseSkipReason, setMseSkipReason] = useState<string | null>(null)

  const hasReference = referenceImageUrl != null
  const hasContour = contourPoints.length >= 3
  const canRunMse = hasReference && hasContour && testImageUrl != null
  const canRunPatchcore = testImageUrl != null
  const analyzing = mseRunning || patchcoreRunning

  const handleReferenceSelected = useCallback((url: string, width: number, height: number) => {
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl)
    setReferenceImageUrl(url)
    setReferenceImageSize({ width, height })
    setContourPoints([])
    setCompareResult(null)
    setMseSkipReason(null)
  }, [referenceImageUrl])

  const handleTestImageLoad = useCallback((url: string, width?: number, height?: number) => {
    if (testImageUrl) URL.revokeObjectURL(testImageUrl)
    setTestImageUrl(url)
    setTestImageSize({ width: width ?? 0, height: height ?? 0 })
    setCompareResult(null)
    setPatchcoreResult(null)
    setMseSkipReason(null)
  }, [testImageUrl])

  const runMseComparison = useCallback(() => {
    if (!referenceImageUrl || !testImageUrl || !hasContour) return
    setMseRunning(true)
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
      setMseRunning(false)
    }
    testImg.onload = run
    refImg.onload = () => { testImg.src = testImageUrl }
    refImg.onerror = () => setMseRunning(false)
    testImg.onerror = () => setMseRunning(false)
    refImg.src = referenceImageUrl
  }, [
    referenceImageUrl,
    testImageUrl,
    contourPoints,
    paddingPx,
    defectThresholdMse,
    ignoreDiffThreshold,
    hasContour,
  ])

  const runPatchcore = useCallback(async () => {
    if (!testImageUrl) return
    setPatchcoreRunning(true)
    setPatchcoreResult(null)
    try {
      const testRes = await fetch(testImageUrl)
      if (!testRes.ok) throw new Error('Не удалось загрузить тестовое изображение')
      const testBlob = await testRes.blob()
      const refBlob = referenceImageUrl ? await (await fetch(referenceImageUrl)).blob() : null
      const result = await analyzePatchcore(
        apiBaseUrl,
        testBlob,
        patchcoreThreshold,
        refBlob
      )
      setPatchcoreResult(result)
    } finally {
      setPatchcoreRunning(false)
    }
  }, [testImageUrl, referenceImageUrl, apiBaseUrl, patchcoreThreshold])

  const handleAnalyze = useCallback(() => {
    setMseSkipReason(null)
    if (!testImageUrl) return

    if (canRunMse) {
      runMseComparison()
    } else {
      if (!hasReference) {
        setMseSkipReason('Сравнение по алгоритму недоступно: не выбран эталон.')
      } else if (!hasContour) {
        setMseSkipReason('Сравнение по алгоритму недоступно: не задан контур на эталоне (нужно ≥3 точки).')
      }
    }

    if (canRunPatchcore) {
      runPatchcore()
    }
  }, [
    testImageUrl,
    canRunMse,
    canRunPatchcore,
    hasReference,
    hasContour,
    runMseComparison,
    runPatchcore,
  ])

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Детектор дефектов этикетки</h1>
        <p className="app__subtitle">
          Загрузите тестовое фото (обязательно) и при необходимости эталон с контуром для двух видов анализа.
        </p>
      </header>

      <div className="app__body">
        <main className="app__main">
          <ReferenceBlock
            referenceImageUrl={referenceImageUrl}
            referenceImageSize={referenceImageSize}
            onReferenceSelected={handleReferenceSelected}
            contourPoints={contourPoints}
            onContourChange={setContourPoints}
            paddingPx={paddingPx}
            onPaddingChange={setPaddingPx}
          />

          <TestBlock testImageUrl={testImageUrl} onTestImageLoad={handleTestImageLoad} />

          <AnalyzeButton
            onClick={handleAnalyze}
            analyzing={analyzing}
            disabled={!testImageUrl}
            disabledReason={!testImageUrl ? 'Загрузите тестовое изображение' : undefined}
          />

          {mseSkipReason && (
            <div className="app__notice app__notice--info" role="status">
              {mseSkipReason}
            </div>
          )}

          <ResultBlocks
            testImageUrl={testImageUrl}
            testImageSize={testImageSize}
            compareResult={compareResult}
            anomalibResult={patchcoreResult}
          />
        </main>

        <SidebarPanel
          defectThresholdMse={defectThresholdMse}
          onDefectThresholdMseChange={setDefectThresholdMse}
          ignoreDiffThreshold={ignoreDiffThreshold}
          onIgnoreDiffThresholdChange={setIgnoreDiffThreshold}
          apiBaseUrl={apiBaseUrl}
          onApiBaseUrlChange={setApiBaseUrl}
          patchcoreThreshold={patchcoreThreshold}
          onPatchcoreThresholdChange={setPatchcoreThreshold}
        />
      </div>
    </div>
  )
}

export default App
