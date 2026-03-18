/**
 * Сравнение изображений: поиск эталонного ROI на тестовом фото и оценка отличий (MSE).
 * Ускорение: поиск на уменьшенной копии, затем уточнение в полном разрешении.
 * Чувствительность: можно игнорировать мелкие отличия (порог по разнице в уровнях серого).
 */

export interface Point {
  x: number
  y: number
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

const MAX_SEARCH_DIM = 360
const COARSE_STEP = 4
const FINE_MARGIN = 6

/**
 * Bounding box контура с учётом padding. Можно ограничить по размерам изображения.
 */
export function getRoiRect(
  points: Point[],
  paddingPx: number,
  clampToSize?: { width: number; height: number }
): Rect {
  if (points.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }
  let minX = points[0].x
  let minY = points[0].y
  let maxX = points[0].x
  let maxY = points[0].y
  for (let i = 1; i < points.length; i++) {
    minX = Math.min(minX, points[i].x)
    minY = Math.min(minY, points[i].y)
    maxX = Math.max(maxX, points[i].x)
    maxY = Math.max(maxY, points[i].y)
  }
  let x = Math.floor(minX - paddingPx)
  let y = Math.floor(minY - paddingPx)
  let width = Math.ceil(maxX - minX + 2 * paddingPx)
  let height = Math.ceil(maxY - minY + 2 * paddingPx)
  if (clampToSize) {
    x = Math.max(0, Math.min(x, clampToSize.width - 1))
    y = Math.max(0, Math.min(y, clampToSize.height - 1))
    const x2 = Math.min(x + width, clampToSize.width)
    const y2 = Math.min(y + height, clampToSize.height)
    width = Math.max(0, x2 - x)
    height = Math.max(0, y2 - y)
  } else {
    x = Math.max(0, x)
    y = Math.max(0, y)
  }
  return { x, y, width, height }
}

/** Точка (x,y) лежит внутри полигона (ray casting). */
function pointInPolygon(x: number, y: number, points: Point[]): boolean {
  const n = points.length
  if (n < 3) return false
  let inside = false
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = points[i].x
    const yi = points[i].y
    const xj = points[j].x
    const yj = points[j].y
    if (yi === yj) continue
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

/** Маска ROI: 255 — пиксель внутри контура, 0 — снаружи. rect в координатах изображения. */
function getContourMask(points: Point[], rect: Rect): Uint8Array {
  const { x: rx, y: ry, width: rw, height: rh } = rect
  const mask = new Uint8Array(rw * rh)
  if (points.length < 3) return mask
  for (let py = 0; py < rh; py++) {
    for (let px = 0; px < rw; px++) {
      const imgX = rx + px + 0.5
      const imgY = ry + py + 0.5
      mask[py * rw + px] = pointInPolygon(imgX, imgY, points) ? 255 : 0
    }
  }
  return mask
}

/**
 * Уменьшить серое изображение (через canvas).
 */
function downscaleGray(
  gray: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  const srcCanvas = document.createElement('canvas')
  srcCanvas.width = srcW
  srcCanvas.height = srcH
  const ctxSrc = srcCanvas.getContext('2d')!
  const imageData = ctxSrc.createImageData(srcW, srcH)
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i]
    imageData.data[i * 4] = v
    imageData.data[i * 4 + 1] = v
    imageData.data[i * 4 + 2] = v
    imageData.data[i * 4 + 3] = 255
  }
  ctxSrc.putImageData(imageData, 0, 0)

  const dstCanvas = document.createElement('canvas')
  dstCanvas.width = dstW
  dstCanvas.height = dstH
  const ctxDst = dstCanvas.getContext('2d')!
  ctxDst.drawImage(srcCanvas, 0, 0, srcW, srcH, 0, 0, dstW, dstH)
  const out = ctxDst.getImageData(0, 0, dstW, dstH)
  const result = new Uint8Array(dstW * dstH)
  for (let i = 0; i < result.length; i++) {
    result[i] = (out.data[i * 4] * 0.299 + out.data[i * 4 + 1] * 0.587 + out.data[i * 4 + 2] * 0.114) | 0
  }
  return result
}

/**
 * Уменьшить патч (выборка пикселей).
 */
function downscalePatch(
  patch: Uint8Array,
  patchW: number,
  patchH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  const result = new Uint8Array(dstW * dstH)
  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(Math.floor((dx + 0.5) * patchW / dstW), patchW - 1)
      const sy = Math.min(Math.floor((dy + 0.5) * patchH / dstH), patchH - 1)
      result[dy * dstW + dx] = patch[sy * patchW + sx]
    }
  }
  return result
}

/** Уменьшить маску (те же индексы, что и у downscalePatch). */
function downscaleMask(
  mask: Uint8Array,
  patchW: number,
  patchH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  const result = new Uint8Array(dstW * dstH)
  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = Math.min(Math.floor((dx + 0.5) * patchW / dstW), patchW - 1)
      const sy = Math.min(Math.floor((dy + 0.5) * patchH / dstH), patchH - 1)
      result[dy * dstW + dx] = mask[sy * patchW + sx] ? 255 : 0
    }
  }
  return result
}

/**
 * Рисуем изображение в canvas и возвращаем ImageData (grayscale для ускорения).
 * Используем натуральные размеры изображения (intrinsic dimensions).
 */
function imageToGrayData(
  img: HTMLImageElement | ImageBitmap,
  canvas: HTMLCanvasElement
): { data: Uint8Array; width: number; height: number } {
  const w = 'naturalWidth' in img && img.naturalWidth > 0 ? img.naturalWidth : img.width
  const h = 'naturalHeight' in img && img.naturalHeight > 0 ? img.naturalHeight : img.height
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0)
  const imageData = ctx.getImageData(0, 0, w, h)
  const gray = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) {
    const r = imageData.data[i * 4]
    const g = imageData.data[i * 4 + 1]
    const b = imageData.data[i * 4 + 2]
    gray[i] = (r * 0.299 + g * 0.587 + b * 0.114) | 0
  }
  return { data: gray, width: w, height: h }
}

/**
 * Вырезать патч из gray data.
 */
function extractPatch(
  gray: Uint8Array,
  imgW: number,
  imgH: number,
  rect: Rect
): Uint8Array | null {
  const { x, y, width, height } = rect
  if (x < 0 || y < 0 || x + width > imgW || y + height > imgH) return null
  const patch = new Uint8Array(width * height)
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const srcIdx = (y + py) * imgW + (x + px)
      patch[py * width + px] = gray[srcIdx]
    }
  }
  return patch
}

/**
 * Карта «гладкости» патча: по каждому пикселю — сила локального градиента (0 = гладкая зона, 255 = резкий край).
 * Дефект (трещина, пятно) появляется на гладком фоне эталона, поэтому подсвечиваем только где гладко + есть отличие.
 */
function computeSmoothMap(patch: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h)
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const i = py * w + px
      const v = patch[i]
      let g = 0
      if (px > 0) g = Math.max(g, Math.abs(v - patch[py * w + (px - 1)]))
      if (px < w - 1) g = Math.max(g, Math.abs(v - patch[py * w + (px + 1)]))
      if (py > 0) g = Math.max(g, Math.abs(v - patch[(py - 1) * w + px]))
      if (py < h - 1) g = Math.max(g, Math.abs(v - patch[(py + 1) * w + px]))
      out[i] = Math.min(255, g)
    }
  }
  return out
}

/**
 * Template matching: найти позицию (tx, ty) в testGray, где патч referencePatch
 * лучше всего совпадает (минимальная MSE). Учитываются только пиксели, где mask[i] != 0.
 */
function findBestMatch(
  referencePatch: Uint8Array,
  patchW: number,
  patchH: number,
  testGray: Uint8Array,
  testW: number,
  testH: number,
  step: number,
  ignoreDiff: number,
  mask: Uint8Array
): { x: number; y: number; mse: number } {
  let bestX = 0
  let bestY = 0
  let bestMse = Infinity
  const maskCount = mask.reduce((s, v) => s + (v ? 1 : 0), 0)
  if (maskCount === 0) return { x: 0, y: 0, mse: Infinity }

  for (let ty = 0; ty <= testH - patchH; ty += step) {
    for (let tx = 0; tx <= testW - patchW; tx += step) {
      let sumSq = 0
      for (let i = 0; i < referencePatch.length; i++) {
        if (!mask[i]) continue
        const py = (i / patchW) | 0
        const px = i % patchW
        const testIdx = (ty + py) * testW + (tx + px)
        let d = referencePatch[i] - testGray[testIdx]
        if (Math.abs(d) > ignoreDiff) sumSq += d * d
      }
      const mse = sumSq / maskCount
      if (mse < bestMse) {
        bestMse = mse
        bestX = tx
        bestY = ty
      }
    }
  }
  return { x: bestX, y: bestY, mse: bestMse }
}

/**
 * MSE между двумя патчами только по пикселям, где mask[i] != 0.
 */
function mseBetween(
  a: Uint8Array,
  b: Uint8Array,
  ignoreDiff: number,
  mask: Uint8Array
): number {
  let sumSq = 0
  let count = 0
  for (let i = 0; i < a.length; i++) {
    if (!mask[i]) continue
    count++
    const d = a[i] - b[i]
    if (Math.abs(d) > ignoreDiff) sumSq += d * d
  }
  return count === 0 ? Infinity : sumSq / count
}

export interface CompareResult {
  defect: boolean
  mse: number
  threshold: number
  matchPosition: { x: number; y: number } | null
  roiSize: { width: number; height: number }
  /** Маска контура: 255 = внутри контура, 0 = снаружи. Размер roiSize.width * roiSize.height. */
  contourMask: Uint8Array | null
  diffMap: Uint8Array | null
  smoothMap: Uint8Array | null
  ignoreDiffUsed: number
  /** время выполнения алгоритма сравнения на клиенте (мс) */
  algo_ms?: number
}

/**
 * Сравнить эталонное изображение (с заданным ROI по контуру и padding)
 * с тестовым изображением. Возвращает результат: брак/нет и MSE.
 *
 * Ускорение: при больших размерах поиск идёт на уменьшенной копии (макс. 360px),
 * затем уточнение в полном разрешении в малой окрестности.
 *
 * ignoreDiffThreshold: разница по яркости (0–255) ниже этого не считается дефектом
 * (мелкие царапины/линии не засчитываются). По умолчанию 8.
 */
export function compareWithReference(
  referenceImage: HTMLImageElement | ImageBitmap,
  testImage: HTMLImageElement | ImageBitmap,
  contourPoints: Point[],
  paddingPx: number,
  defectThresholdMse: number,
  ignoreDiffThreshold: number = 8
): CompareResult {
  const canvas = document.createElement('canvas')
  const refGray = imageToGrayData(referenceImage, canvas)
  const testGray = imageToGrayData(testImage, canvas)
  const testW = testGray.width
  const testH = testGray.height

  const roi = getRoiRect(contourPoints, paddingPx, {
    width: refGray.width,
    height: refGray.height,
  })
  if (roi.width <= 0 || roi.height <= 0) {
    return {
      defect: true,
      mse: Infinity,
      threshold: defectThresholdMse,
      matchPosition: null,
      roiSize: { width: 0, height: 0 },
      contourMask: null,
      diffMap: null,
      smoothMap: null,
      ignoreDiffUsed: ignoreDiffThreshold,
    }
  }

  const contourMask = getContourMask(contourPoints, roi)
  const maskCount = contourMask.reduce((s, v) => s + (v ? 1 : 0), 0)
  if (maskCount === 0) {
    return {
      defect: true,
      mse: Infinity,
      threshold: defectThresholdMse,
      matchPosition: null,
      roiSize: { width: roi.width, height: roi.height },
      contourMask,
      diffMap: null,
      smoothMap: null,
      ignoreDiffUsed: ignoreDiffThreshold,
    }
  }

  const referencePatch = extractPatch(
    refGray.data,
    refGray.width,
    refGray.height,
    roi
  )
  if (!referencePatch) {
    return {
      defect: true,
      mse: Infinity,
      threshold: defectThresholdMse,
      matchPosition: null,
      roiSize: { width: roi.width, height: roi.height },
      contourMask: null,
      diffMap: null,
      smoothMap: null,
      ignoreDiffUsed: ignoreDiffThreshold,
    }
  }

  const refW = refGray.width
  const refH = refGray.height
  const sameSize = refW === testW && refH === testH

  // Размер патча и маска в координатах теста (при разном разрешении — масштабируем из эталона в тест)
  let resultRoiW: number
  let resultRoiH: number
  let resultPatch: Uint8Array
  let resultMask: Uint8Array
  let matchX: number
  let matchY: number

  if (sameSize) {
    resultRoiW = roi.width
    resultRoiH = roi.height
    resultPatch = referencePatch
    resultMask = contourMask
    matchX = roi.x
    matchY = roi.y
  } else {
    const scaleToTestX = testW / refW
    const scaleToTestY = testH / refH
    resultRoiW = Math.max(1, Math.round(roi.width * scaleToTestX))
    resultRoiH = Math.max(1, Math.round(roi.height * scaleToTestY))
    resultPatch = downscalePatch(referencePatch, roi.width, roi.height, resultRoiW, resultRoiH)
    resultMask = downscaleMask(contourMask, roi.width, roi.height, resultRoiW, resultRoiH)

    const maskCountScaled = resultMask.reduce((s, v) => s + (v ? 1 : 0), 0)
    if (maskCountScaled === 0) {
      return {
        defect: true,
        mse: Infinity,
        threshold: defectThresholdMse,
        matchPosition: null,
        roiSize: { width: resultRoiW, height: resultRoiH },
        contourMask: resultMask,
        diffMap: null,
        smoothMap: null,
        ignoreDiffUsed: ignoreDiffThreshold,
      }
    }

    const maxDim = Math.max(testW, testH)
    if (maxDim > MAX_SEARCH_DIM) {
      const scale = MAX_SEARCH_DIM / maxDim
      const smallW = Math.max(1, Math.round(testW * scale))
      const smallH = Math.max(1, Math.round(testH * scale))
      const patchSmallW = Math.max(1, Math.round(resultRoiW * scale))
      const patchSmallH = Math.max(1, Math.round(resultRoiH * scale))

      const testSmall = downscaleGray(testGray.data, testW, testH, smallW, smallH)
      const patchSmall = downscalePatch(
        resultPatch,
        resultRoiW,
        resultRoiH,
        patchSmallW,
        patchSmallH
      )
      const maskSmall = downscaleMask(
        resultMask,
        resultRoiW,
        resultRoiH,
        patchSmallW,
        patchSmallH
      )

      const coarse = findBestMatch(
        patchSmall,
        patchSmallW,
        patchSmallH,
        testSmall,
        smallW,
        smallH,
        COARSE_STEP,
        ignoreDiffThreshold,
        maskSmall
      )
      matchX = Math.round(coarse.x / scale)
      matchY = Math.round(coarse.y / scale)
      matchX = Math.max(0, Math.min(matchX, testW - resultRoiW))
      matchY = Math.max(0, Math.min(matchY, testH - resultRoiH))
    } else {
      const coarse = findBestMatch(
        resultPatch,
        resultRoiW,
        resultRoiH,
        testGray.data,
        testW,
        testH,
        COARSE_STEP,
        ignoreDiffThreshold,
        resultMask
      )
      matchX = coarse.x
      matchY = coarse.y
    }
  }

  let fineMse = Infinity
  let fineX = matchX
  let fineY = matchY
  const margin = FINE_MARGIN
  for (let dy = -margin; dy <= margin; dy++) {
    for (let dx = -margin; dx <= margin; dx++) {
      const tx = matchX + dx
      const ty = matchY + dy
      if (tx < 0 || ty < 0 || tx + resultRoiW > testW || ty + resultRoiH > testH) continue
      const testPatch = extractPatch(testGray.data, testW, testH, {
        x: tx,
        y: ty,
        width: resultRoiW,
        height: resultRoiH,
      })
      if (testPatch) {
        const m = mseBetween(resultPatch, testPatch, ignoreDiffThreshold, resultMask)
        if (m < fineMse) {
          fineMse = m
          fineX = tx
          fineY = ty
        }
      }
    }
  }

  const finalTestPatch = extractPatch(testGray.data, testW, testH, {
    x: fineX,
    y: fineY,
    width: resultRoiW,
    height: resultRoiH,
  })
  if (!finalTestPatch) {
    return {
      defect: fineMse > defectThresholdMse,
      mse: fineMse,
      threshold: defectThresholdMse,
      matchPosition: { x: fineX, y: fineY },
      roiSize: { width: resultRoiW, height: resultRoiH },
      contourMask: null,
      diffMap: null,
      smoothMap: null,
      ignoreDiffUsed: ignoreDiffThreshold,
    }
  }

  const diffMap = (() => {
    const d = new Uint8Array(resultPatch.length)
    for (let i = 0; i < d.length; i++) {
      if (!resultMask[i]) continue
      d[i] = Math.min(255, Math.abs(resultPatch[i] - finalTestPatch[i]))
    }
    return d
  })()
  const smoothMap = (() => {
    const sm = computeSmoothMap(resultPatch, resultRoiW, resultRoiH)
    for (let i = 0; i < sm.length; i++) {
      if (!resultMask[i]) sm[i] = 255
    }
    return sm
  })()

  // Брак: средняя MSE выше порога ИЛИ заметная доля пикселей отличается (пятно/царапина)
  const strongDiffThreshold = 32
  const veryStrongDiff = 70 // один такой пиксель — уже брак (явное пятно на этикетке)
  let strongDiffCount = 0
  let hasVeryStrong = false
  for (let i = 0; i < diffMap.length; i++) {
    if (!resultMask[i]) continue
    if (diffMap[i] >= veryStrongDiff) hasVeryStrong = true
    if (diffMap[i] >= strongDiffThreshold) strongDiffCount++
  }
  const maskedPixels = resultMask.reduce((s, v) => s + (v ? 1 : 0), 0)
  const strongDiffFraction = maskedPixels > 0 ? strongDiffCount / maskedPixels : 0
  const hasLocalDefect =
    hasVeryStrong || strongDiffFraction >= 0.0008 // 0.08% пикселей с отличием ≥32 или хотя бы один ≥70

  const defect =
    fineMse > defectThresholdMse || hasLocalDefect

  return {
    defect,
    mse: fineMse,
    threshold: defectThresholdMse,
    matchPosition: { x: fineX, y: fineY },
    roiSize: { width: resultRoiW, height: resultRoiH },
    contourMask: resultMask,
    diffMap,
    smoothMap,
    ignoreDiffUsed: ignoreDiffThreshold,
  }
}
