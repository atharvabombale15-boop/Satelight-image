import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE || '/api'

function fileToDataUrl(file, setter) {
  if (!file) {
    setter('')
    return
  }
  const reader = new FileReader()
  reader.onload = () => setter(reader.result || '')
  reader.readAsDataURL(file)
}

function App() {
  const [t1File, setT1File] = useState(null)
  const [t2File, setT2File] = useState(null)
  const [t1Raw, setT1Raw] = useState(null)
  const [t2Raw, setT2Raw] = useState(null)
  const [t1Preview, setT1Preview] = useState('')
  const [t2Preview, setT2Preview] = useState('')
  const [sliderValue, setSliderValue] = useState(50)
  const [threshold, setThreshold] = useState(0.2)
  const [thresholdMode, setThresholdMode] = useState('fixed')
  const [percentile, setPercentile] = useState(90)
  const [modelArch, setModelArch] = useState('siamese_unet')
  const [imageSize, setImageSize] = useState(256)
  const [imageSizeOptions, setImageSizeOptions] = useState([256, 384, 512])
  const [useTiling, setUseTiling] = useState(true)
  const [align, setAlign] = useState(true)
  const [minArea, setMinArea] = useState(64)
  const [multiscale, setMultiscale] = useState(false)
  const [fastMode, setFastMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [batchFile, setBatchFile] = useState(null)
  const [batchResult, setBatchResult] = useState(null)

  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch(`${API_BASE}/config`)
        if (!response.ok) return
        const cfg = await response.json()
        if (typeof cfg.default_threshold === 'number') setThreshold(cfg.default_threshold)
        if (typeof cfg.default_threshold_mode === 'string') setThresholdMode(cfg.default_threshold_mode)
        if (typeof cfg.default_percentile === 'number') setPercentile(cfg.default_percentile)
        if (typeof cfg.model_arch === 'string') setModelArch(cfg.model_arch)
        if (typeof cfg.default_image_size === 'number') setImageSize(cfg.default_image_size)
        if (Array.isArray(cfg.image_sizes)) setImageSizeOptions(cfg.image_sizes)
        if (typeof cfg.default_align === 'boolean') setAlign(cfg.default_align)
        if (typeof cfg.default_min_area === 'number') setMinArea(cfg.default_min_area)
        if (typeof cfg.default_multiscale === 'boolean') setMultiscale(cfg.default_multiscale)
        if (typeof cfg.default_fast_mode === 'boolean') setFastMode(cfg.default_fast_mode)
      } catch (err) {
        // Fallback to local defaults if config fetch fails
      }
    }
    loadConfig()
  }, [])

  const modelLabel = useMemo(() => {
    const raw = (modelArch || '').toString()
    const normalized = raw.replace(/_/g, ' ').toLowerCase()
    if (normalized.includes('siamese')) return 'Siamese U-Net'
    if (normalized.includes('changeformer')) return 'ChangeFormer Lite'
    return raw || 'Model'
  }, [modelArch])

  const resizeImageFile = async (file, maxDim) => {
    if (!file) return null
    const bitmap = await createImageBitmap(file)
    const { width, height } = bitmap
    const largest = Math.max(width, height)
    if (largest <= maxDim) {
      return file
    }
    const scale = maxDim / largest
    const targetW = Math.round(width * scale)
    const targetH = Math.round(height * scale)
    const canvas = document.createElement('canvas')
    canvas.width = targetW
    canvas.height = targetH
    const ctx = canvas.getContext('2d')
    ctx.drawImage(bitmap, 0, 0, targetW, targetH)
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.9)
    )
    return new File([blob], file.name.replace(/\.\w+$/, '.jpg'), {
      type: 'image/jpeg',
    })
  }

  const processImage = async (file, isFast) => {
    const maxDim = isFast ? 1024 : 2048
    return resizeImageFile(file, maxDim)
  }

  useEffect(() => {
    const reprocess = async () => {
      if (t1Raw) {
        const processed = await processImage(t1Raw, fastMode)
        setT1File(processed)
        fileToDataUrl(processed, setT1Preview)
      }
      if (t2Raw) {
        const processed = await processImage(t2Raw, fastMode)
        setT2File(processed)
        fileToDataUrl(processed, setT2Preview)
      }
    }
    reprocess()
  }, [fastMode])

  const maskUrl = useMemo(() => {
    if (!result?.mask_png_base64) return ''
    return `data:image/png;base64,${result.mask_png_base64}`
  }, [result])

  const overlayUrl = useMemo(() => {
    if (!result?.overlay_png_base64) return ''
    return `data:image/png;base64,${result.overlay_png_base64}`
  }, [result])

  const confidenceUrl = useMemo(() => {
    if (!result?.confidence_png_base64) return ''
    return `data:image/png;base64,${result.confidence_png_base64}`
  }, [result])


  const handleT1Change = async (event) => {
    const file = event.target.files[0]
    setT1Raw(file || null)
    if (!file) {
      setT1File(null)
      setT1Preview('')
      return
    }
    const processed = await processImage(file, fastMode)
    setT1File(processed)
    fileToDataUrl(processed, setT1Preview)
  }

  const handleT2Change = async (event) => {
    const file = event.target.files[0]
    setT2Raw(file || null)
    if (!file) {
      setT2File(null)
      setT2Preview('')
      return
    }
    const processed = await processImage(file, fastMode)
    setT2File(processed)
    fileToDataUrl(processed, setT2Preview)
  }

  const handleRun = async () => {
    if (!t1File || !t2File) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const form = new FormData()
      form.append('t1', t1File)
      form.append('t2', t2File)
      form.append('threshold', threshold.toString())
      form.append('threshold_mode', thresholdMode)
      form.append('percentile', percentile.toString())
      form.append('image_size', imageSize.toString())
      form.append('use_tiling', useTiling ? 'true' : 'false')
      form.append('align', align ? 'true' : 'false')
      form.append('min_area', minArea.toString())
      form.append('multiscale', multiscale ? 'true' : 'false')
      form.append('fast_mode', fastMode ? 'true' : 'false')

      const response = await fetch(`${API_BASE}/infer`, {
        method: 'POST',
        body: form,
      })
      if (!response.ok) {
        throw new Error('Inference request failed.')
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchRun = async () => {
    if (!batchFile) return
    setLoading(true)
    setError('')
    setBatchResult(null)
    try {
      const form = new FormData()
      form.append('archive', batchFile)
      form.append('threshold', threshold.toString())
      form.append('threshold_mode', thresholdMode)
      form.append('percentile', percentile.toString())
      form.append('image_size', imageSize.toString())
      form.append('use_tiling', useTiling ? 'true' : 'false')
      form.append('align', align ? 'true' : 'false')
      form.append('min_area', minArea.toString())
      form.append('multiscale', multiscale ? 'true' : 'false')

      const response = await fetch(`${API_BASE}/batch`, {
        method: 'POST',
        body: form,
      })
      if (!response.ok) {
        throw new Error('Batch request failed.')
      }
      const data = await response.json()
      setBatchResult(data)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <h1>Urban Change Detection</h1>
          <p className="subtitle">
            Analyze temporal satellite pairs to surface structural change.
          </p>
        </div>
        <div className="hero-pills">
          <span>Model: {modelLabel}</span>
          <span>Resolution: {imageSize}x{imageSize}</span>
        </div>
      </header>

      <section className="grid">
        <div className="panel">
          <h2>Input Console</h2>
          <div className="upload-grid">
            <label className="upload-card">
              <span className="label">T1 Image</span>
              <span className="hint">Older satellite capture</span>
              <input type="file" accept="image/*" onChange={handleT1Change} />
            </label>
            <label className="upload-card">
              <span className="label">T2 Image</span>
              <span className="hint">Newer satellite capture</span>
              <input type="file" accept="image/*" onChange={handleT2Change} />
            </label>
            <div className="status-card">
              <h3>System</h3>
              <p>
                {loading
                  ? 'Running inference...'
                  : result
                    ? 'Inference complete. Review outputs below.'
                    : t1File && t2File
                      ? 'Ready to detect change.'
                      : 'Awaiting imagery to activate inference.'}
              </p>
              <button className="run-button" onClick={handleRun} disabled={!t1File || !t2File || loading}>
                {loading ? 'Processing...' : 'Detect Change'}
              </button>
              {error ? <p className="error">{error}</p> : null}
            </div>
          </div>
        </div>

        <aside className="panel settings">
          <h2>Model Settings</h2>
          <div className="setting">
            <label>Change Threshold: {threshold.toFixed(2)}</label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
          </div>
          <div className="setting">
            <label>Threshold Mode</label>
            <select value={thresholdMode} onChange={(e) => setThresholdMode(e.target.value)}>
              <option value="fixed">Fixed</option>
              <option value="otsu">Otsu</option>
              <option value="percentile">Percentile</option>
            </select>
          </div>
          {thresholdMode === 'percentile' ? (
            <div className="setting">
              <label>Percentile: {percentile}</label>
              <input
                type="range"
                min="50"
                max="99"
                step="1"
                value={percentile}
                onChange={(e) => setPercentile(parseInt(e.target.value, 10))}
              />
            </div>
          ) : null}
          <div className="setting">
            <label>Inference Resolution</label>
            <select value={imageSize} onChange={(e) => setImageSize(parseInt(e.target.value, 10))} disabled={fastMode}>
              {imageSizeOptions.map((size) => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>
          <div className="setting checkbox">
            <label>
              <input
                type="checkbox"
                checked={align}
                onChange={(e) => setAlign(e.target.checked)}
              />
              Align T2 to T1 before inference
            </label>
          </div>
          <div className="setting checkbox">
            <label>
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(e) => setFastMode(e.target.checked)}
              />
              Fast mode (lower fidelity, faster)
            </label>
          </div>
          <div className="setting checkbox">
            <label>
              <input
                type="checkbox"
                checked={useTiling}
                onChange={(e) => setUseTiling(e.target.checked)}
                disabled={fastMode}
              />
              Use tiling for large images
            </label>
          </div>
          <div className="setting checkbox">
            <label>
              <input
                type="checkbox"
                checked={multiscale}
                onChange={(e) => setMultiscale(e.target.checked)}
                disabled={fastMode}
              />
              Multi-scale inference (256 + 512 blend)
            </label>
          </div>
          <div className="setting">
            <label>Remove tiny speckles (min area)</label>
            <input
              type="number"
              min="0"
              step="1"
              value={minArea}
              onChange={(e) => setMinArea(parseInt(e.target.value || '0', 10))}
            />
          </div>
          <p className="muted">Higher resolution improves detail but increases compute time.</p>
        </aside>
      </section>

      <section className="panel">
        <h2>Input Preview</h2>
        <div className="preview-grid">
          <div className="preview-card">
            {t1Preview ? <img src={t1Preview} alt="T1 preview" /> : <p className="muted">No T1 image.</p>}
            <span>T1</span>
          </div>
          <div className="preview-card">
            {t2Preview ? <img src={t2Preview} alt="T2 preview" /> : <p className="muted">No T2 image.</p>}
            <span>T2</span>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Batch Processing</h2>
        <p className="muted">Upload a ZIP with folders `t1/` and `t2/` containing matching filenames.</p>
        <div className="batch-row">
          <input
            type="file"
            accept=".zip"
            onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
          />
          <button className="run-button" onClick={handleBatchRun} disabled={!batchFile || loading}>
            {loading ? 'Processing...' : 'Process ZIP'}
          </button>
        </div>
        {batchResult ? (
          <div className="batch-results">
            <p className="muted">Processed {batchResult.count} pairs.</p>
            <div className="download-grid">
              <a
                className="download-button"
                href={`data:text/csv;base64,${batchResult.csv_base64}`}
                download="batch_report.csv"
              >
                Download CSV
              </a>
              <a
                className="download-button"
                href={`data:text/html;base64,${batchResult.html_base64}`}
                download="batch_report.html"
              >
                Download HTML
              </a>
            </div>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <h2>Before / After Slider</h2>
        {t1Preview && t2Preview ? (
          <div className="slider-block">
            <div className="slider-wrap">
              <span className="tag left">T1</span>
              <span className="tag right">T2</span>
              <img src={t1Preview} alt="T1" />
              <img
                className="after"
                src={t2Preview}
                alt="T2"
                style={{ clipPath: `inset(0 0 0 ${sliderValue}%)` }}
              />
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={sliderValue}
              onChange={(e) => setSliderValue(parseInt(e.target.value, 10))}
            />
            <p className="muted">T2 visibility: {sliderValue}%</p>
          </div>
        ) : (
          <p className="muted">Upload both images to activate the slider.</p>
        )}
      </section>

      {result ? (
        <>
          <section className="panel">
            <h2>Results</h2>
            <div className="metric-grid">
              <div className="metric-card">
                <h3>Urban Change</h3>
                <p>{result.change_percentage.toFixed(2)}%</p>
              </div>
              <div className="metric-card">
                <h3>Threshold</h3>
                <p>{result.threshold.toFixed(2)}</p>
              </div>
              <div className="metric-card">
                <h3>Pixels Changed</h3>
                <p>{result.pixels_changed}</p>
              </div>
            </div>
          </section>

          <section className="panel">
            <h2>Outputs</h2>
            <p className="legend">
              Confidence Map: brighter = higher change probability.
            </p>
            <div className="preview-grid">
              <div className="preview-card">
                <img src={maskUrl} alt="Change mask" />
                <span>Change Mask</span>
              </div>
              <div className="preview-card">
                <img src={overlayUrl} alt="Overlay" />
                <span>Overlay</span>
              </div>
              <div className="preview-card">
                <img src={confidenceUrl} alt="Confidence map" />
                <span>Confidence Map</span>
              </div>
            </div>
            {result.alignment ? (
              <div className="meta">
                <p className="muted">
                  Alignment: {result.alignment.aligned ? 'enabled' : 'skipped'}{' '}
                  {result.alignment.debug?.inliers !== undefined
                    ? `(inliers ${result.alignment.debug.inliers}/${result.alignment.debug.num_matches})`
                    : ''}
                </p>
                {typeof result.alignment_score === 'number' ? (
                  <div className="quality">
                    <span>Alignment Quality</span>
                    <div className="quality-bar">
                      <div
                        className="quality-fill"
                        style={{ width: `${Math.round(result.alignment_score * 100)}%` }}
                      />
                    </div>
                    <span>{Math.round(result.alignment_score * 100)}%</span>
                  </div>
                ) : null}
                <p className="muted">
                  Post-process: removed {result.postprocess?.removed ?? 0} small components
                </p>
              </div>
            ) : null}
          </section>

          <section className="panel">
            <h2>Downloads</h2>
            <div className="download-grid">
              <a className="download-button" href={maskUrl} download="change_mask.png">
                Download Change Mask
              </a>
              <a className="download-button" href={overlayUrl} download="overlay.png">
                Download Overlay
              </a>
              <a className="download-button" href={confidenceUrl} download="confidence_map.png">
                Download Confidence Map
              </a>
            </div>
          </section>
        </>
      ) : (
        <section className="panel">
          <h2>Mission Brief</h2>
          <div className="mission-grid">
            <div className="mission-card">
              <h3>Step 1</h3>
              <p>Upload a T1 image from the earlier capture window.</p>
            </div>
            <div className="mission-card">
              <h3>Step 2</h3>
              <p>Upload the corresponding T2 image from the latest capture.</p>
            </div>
            <div className="mission-card">
              <h3>Step 3</h3>
              <p>Run inference and review the change masks for urban growth signals.</p>
            </div>
          </div>
        </section>
      )}
      <footer className="footer">
        Tip: For best results, ensure both images are aligned and captured under similar lighting.
      </footer>
    </div>
  )
}

export default App
