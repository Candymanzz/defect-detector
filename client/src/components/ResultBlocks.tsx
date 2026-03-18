import { DefectHighlight } from './DefectHighlight'
import type { CompareResult } from '../lib/compare'
import type { PatchcoreAnalyzeResult } from '../api/patchcore'

interface ResultBlocksProps {
  testImageUrl: string | null
  testImageSize: { width: number; height: number }
  compareResult: CompareResult | null
  anomalibResult: PatchcoreAnalyzeResult | null
}

/** Совмещённый вывод: брак, если хотя бы один из методов сработал */
function getCombinedResult(
  compareResult: CompareResult | null,
  anomalibResult: PatchcoreAnalyzeResult | null
) {
  if (!compareResult && !anomalibResult) return null
  const mseDefect = compareResult?.defect ?? null
  const patchcoreDefect =
    anomalibResult != null && !anomalibResult.error
      ? anomalibResult.defect
      : null
  const finalDefect = (mseDefect ?? false) || (patchcoreDefect ?? false)
  const note =
    mseDefect != null && patchcoreDefect != null && mseDefect !== patchcoreDefect
      ? 'Классическое сравнение и Patchcore расходятся: проверьте зону по теплокарте и контуру.'
      : ''
  return { finalDefect, mseDefect, patchcoreDefect, note }
}

export function ResultBlocks({
  testImageUrl,
  testImageSize,
  compareResult,
  anomalibResult,
}: ResultBlocksProps) {
  const combined = getCombinedResult(compareResult, anomalibResult)

  return (
    <>
      {anomalibResult && (
        <div className="result-block result-block--anomalib">
          {anomalibResult.error ? (
            <>
              <div className="result result--defect">
                <span className="result__label">Patchcore:</span>
                <span className="result__value">Ошибка запроса</span>
              </div>
              <p className="result-block__message">
                {anomalibResult.message}
                {anomalibResult.message?.includes('fetch') &&
                  ' — проверьте, что бэкенд запущен и доступен.'}
              </p>
            </>
          ) : (
            <>
              <div className={`result ${anomalibResult.defect ? 'result--defect' : 'result--ok'}`}>
                <span className="result__label">Patchcore:</span>
                <span className="result__value">{anomalibResult.defect ? 'Брак' : 'Нет брака'}</span>
              </div>
              <p className="result-block__mse">
                score (0–1) = {anomalibResult.score.toFixed(4)}
                {anomalibResult.raw_score != null && (
                  <> · сырой = {Number(anomalibResult.raw_score).toFixed(2)}</>
                )}{' '}
                (порог {anomalibResult.threshold} →{' '}
                {anomalibResult.defect ? 'брак' : 'норма'})
              </p>
              {(anomalibResult.nn_inference_ms != null || anomalibResult.total_ms != null) && (
                <p className="result-block__mse">
                  Время Patchcore: нейронка {anomalibResult.nn_inference_ms ?? '—'} мс
                  {anomalibResult.postprocess_ms != null && (
                    <> · постобработка {anomalibResult.postprocess_ms} мс</>
                  )}
                  {anomalibResult.total_ms != null && <> · всего {anomalibResult.total_ms} мс</>}
                </p>
              )}
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
            </>
          )}
        </div>
      )}

      {compareResult && (
        <div className="result-block">
          <div className={`result ${compareResult.defect ? 'result--defect' : 'result--ok'}`}>
            <span className="result__label">Результат (MSE):</span>
            <span className="result__value">
              {compareResult.defect ? 'Брак' : 'Нет брака'}
            </span>
          </div>
          <p className="result-block__mse">
            MSE = {compareResult.mse.toFixed(1)} (порог {compareResult.threshold})
          </p>
          {compareResult.algo_ms != null && (
            <p className="result-block__mse">Время алгоритма (MSE): {compareResult.algo_ms} мс</p>
          )}
          {testImageUrl && compareResult.matchPosition && (
            <DefectHighlight
              testImageUrl={testImageUrl}
              testImageSize={testImageSize}
              compareResult={compareResult}
            />
          )}
        </div>
      )}

      {combined && (
        <div className="result-block result-block--combined">
          {/* <div className={`result ${combined.finalDefect ? 'result--ok' : 'result--defect'}`}>
            <span className="result__label">Совмещённый вывод:</span>
            <span className="result__value">
              {combined.finalDefect ? 'норма' : 'брак'}
            </span>
          </div> */}
          <p className="result-block__mse">
            MSE →{' '}
            {combined.mseDefect == null ? 'нет данных' : combined.mseDefect ? 'брак' : 'норма'}
            {' · '}Patchcore →{' '}
            {combined.patchcoreDefect == null
              ? 'нет данных'
              : combined.patchcoreDefect
                ? 'норма'
                : 'брак'}
          </p>
          {/* {combined.note && (
            <p className="result-block__message">{combined.note}</p>
          )} */}
        </div>
      )}
    </>
  )
}
