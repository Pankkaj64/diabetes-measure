import { useState, useEffect } from 'react'

// Types
interface DatasetInfo {
  original_shape: [number, number]
  shape_after_dedup: [number, number]
  zero_counts: Record<string, number>
  summary_statistics: Record<string, Record<string, number>>
  features_used: string[]
  target: string
  train_size: number
  test_size: number
  smote_train_size: number
  outcome_distribution: {
    no_diabetes: number
    diabetes: number
  }
  imputation_defaults?: Record<string, number>
}

interface SamplePatient {
  id: number
  pregnancies: number
  glucose: number
  skin_thickness: number
  bmi: number
  age: number
  actual_outcome: number
  actual_outcome_label: string
}

interface ModelMetric {
  model: string
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  specificity: number
  confusion_matrix: [[number, number], [number, number]]
  best_params: Record<string, any>
  classification_report: Record<string, any>
}

interface PlotItem {
  filename: string
  title: string
  url: string
}

interface PredictionItem {
  model_name: string
  prediction: number
  probability_no_diabetes: number
  probability_diabetes: number
  risk_level: string
}

interface ShapItem {
  feature: string
  value: number
  shap_value: number
}

interface PredictionResponse {
  is_raw: boolean
  raw_inputs: Record<string, number> | null
  quantile_inputs: Record<string, number>
  predictions: PredictionItem[]
  shap_explanation: ShapItem[]
  base_value: number
}

interface PatientParams {
  pregnancies: number
  glucose: number
  skin_thickness: number
  bmi: number
  age: number
}

interface ClinicalPreset {
  label: string
  description: string
  tier: 'low' | 'moderate' | 'high' | 'neutral'
  params: PatientParams
}

// Reference clinical profiles spanning the dataset's risk spectrum.
type TabId = 'overview' | 'predict' | 'models' | 'evaluation' | 'plots' | 'dataset'

interface NavGroup {
  title: string
  items: { id: TabId; label: string; icon: IconName }[]
}

/* Grouped by workflow stage: see the pipeline, score a patient,
   judge the models, then inspect the underlying data. */
const NAV_GROUPS: NavGroup[] = [
  {
    title: 'Overview',
    items: [
      { id: 'overview', label: 'Dashboard', icon: 'dashboard' },
      { id: 'predict', label: 'Risk Calculator', icon: 'calculator' },
    ],
  },
  {
    title: 'Model Performance',
    items: [
      { id: 'models', label: 'Model Comparison', icon: 'compare' },
      { id: 'evaluation', label: 'Evaluation & Results', icon: 'evaluate' },
    ],
  },
  {
    title: 'Data & Analysis',
    items: [
      { id: 'plots', label: 'Analytics & Plots', icon: 'gallery' },
      { id: 'dataset', label: 'Dataset & Pipeline', icon: 'database' },
    ],
  },
]

const CLINICAL_PRESETS: ClinicalPreset[] = [
  {
    label: 'Healthy Adult',
    tier: 'low',
    description: 'Normal fasting glucose, healthy BMI',
    params: { pregnancies: 1, glucose: 85, skin_thickness: 20, bmi: 22.5, age: 25 },
  },
  {
    label: 'Borderline / Prediabetic',
    tier: 'moderate',
    description: 'Impaired fasting glucose, overweight',
    params: { pregnancies: 3, glucose: 118, skin_thickness: 30, bmi: 28.4, age: 38 },
  },
  {
    label: 'High Risk Profile',
    tier: 'high',
    description: 'Hyperglycemic, obese, advanced age',
    params: { pregnancies: 8, glucose: 175, skin_thickness: 45, bmi: 42.8, age: 58 },
  },
  {
    label: 'Dataset Mean Patient',
    tier: 'neutral',
    description: 'Average across all deduplicated records',
    params: { pregnancies: 3, glucose: 121, skin_thickness: 23, bmi: 32.3, age: 29 },
  },
]

// sklearn's classification_report keys the two outcome classes as "0" / "1".
type IconName =
  | 'dashboard' | 'calculator' | 'compare' | 'evaluate' | 'gallery' | 'database'
  | 'logo' | 'check' | 'alert' | 'offline' | 'close' | 'menu' | 'refresh'
  | 'patients' | 'cpu' | 'target' | 'layers' | 'inbox' | 'spark'

/* Inline SVG keeps icon color bound to currentColor, so nav items,
   stat cards and empty states all inherit their own state color. */
function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const p = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    'aria-hidden': true,
  }
  switch (name) {
    case 'dashboard':
      return <svg {...p}><rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></svg>
    case 'calculator':
      return <svg {...p}><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
    case 'compare':
      return <svg {...p}><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></svg>
    case 'evaluate':
      return <svg {...p}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></svg>
    case 'gallery':
      return <svg {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>
    case 'database':
      return <svg {...p}><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>
    case 'logo':
      return <svg {...p}><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
    case 'check':
      return <svg {...p}><polyline points="20 6 9 17 4 12" /></svg>
    case 'alert':
      return <svg {...p}><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>
    case 'offline':
      return <svg {...p}><line x1="1" y1="1" x2="23" y2="23" /><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55" /><path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39" /><path d="M10.71 5.05A16 16 0 0 1 22.58 9" /><path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88" /><line x1="12" y1="20" x2="12.01" y2="20" /></svg>
    case 'close':
      return <svg {...p}><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
    case 'menu':
      return <svg {...p}><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" /></svg>
    case 'refresh':
      return <svg {...p}><polyline points="23 4 23 10 17 10" /><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" /></svg>
    case 'patients':
      return <svg {...p}><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 0 0-3-3.87" /></svg>
    case 'cpu':
      return <svg {...p}><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /></svg>
    case 'target':
      return <svg {...p}><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></svg>
    case 'layers':
      return <svg {...p}><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></svg>
    case 'inbox':
      return <svg {...p}><polyline points="22 12 16 12 14 15 10 15 8 12 2 12" /><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" /></svg>
    case 'spark':
      return <svg {...p}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg>
  }
}

// sklearn emits float-valued class keys ("0.0" / "1.0") for this pipeline,
// so normalise before looking the label up.
const CLASS_LABELS: Record<string, string> = {
  '0': 'Class 0 — Non-Diabetic',
  '1': 'Class 1 — Diabetic',
  'macro avg': 'Macro Average',
  'weighted avg': 'Weighted Average',
}

function classLabel(key: string): string {
  const numeric = Number(key)
  const lookup = Number.isNaN(numeric) ? key : String(numeric)
  return CLASS_LABELS[lookup] ?? key
}

/* Every figure below is derived from the 2x2 confusion matrix, so the
   evaluation view needs no extra endpoint. cm = [[TN, FP], [FN, TP]]. */
interface ClinicalMetrics {
  tn: number; fp: number; fn: number; tp: number
  total: number
  sensitivity: number
  specificity: number
  ppv: number
  npv: number
  balancedAccuracy: number
  mcc: number
  youden: number
}

function clinicalMetrics(cm: [[number, number], [number, number]]): ClinicalMetrics {
  const [[tn, fp], [fn, tp]] = cm
  const safe = (num: number, den: number) => (den === 0 ? 0 : num / den)

  const sensitivity = safe(tp, tp + fn)
  const specificity = safe(tn, tn + fp)
  const mccDen = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

  return {
    tn, fp, fn, tp,
    total: tn + fp + fn + tp,
    sensitivity,
    specificity,
    ppv: safe(tp, tp + fp),
    npv: safe(tn, tn + fn),
    balancedAccuracy: (sensitivity + specificity) / 2,
    mcc: mccDen === 0 ? 0 : (tp * tn - fp * fn) / mccDen,
    youden: sensitivity + specificity - 1,
  }
}

const RANK_METRIC_LABELS = {
  f1_score: 'F1-Score',
  recall: 'Recall',
  accuracy: 'Accuracy',
  precision: 'Precision',
} as const

type RankMetric = keyof typeof RANK_METRIC_LABELS

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

function EmptyState({ icon, title, message }: { icon: IconName; title: string; message: string }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon"><Icon name={icon} size={20} /></div>
      <h3>{title}</h3>
      <p>{message}</p>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('predict')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)

  // Data states from backend
  const [models, setModels] = useState<ModelMetric[]>([])
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [samplePatients, setSamplePatients] = useState<SamplePatient[]>([])
  const [plots, setPlots] = useState<PlotItem[]>([])
  const [loadingInitial, setLoadingInitial] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [retrying, setRetrying] = useState(false)

  // Input Mode: 'clinical' (real physiological units from dataset) vs 'quantile' ([0.0 - 1.0])
  const [inputMode, setInputMode] = useState<'clinical' | 'quantile'>('clinical')

  // Patient inputs for Clinical Mode (Units: mg/dL, kg/m², years)
  // Default to Patient #1 from diabetes-dataset (1).csv
  const [clinicalParams, setClinicalParams] = useState<PatientParams>({
    pregnancies: 2,
    glucose: 138,
    skin_thickness: 35,
    bmi: 33.6,
    age: 47,
  })

  // Patient inputs for Quantile Mode ([0.0 - 1.0])
  const [quantileParams, setQuantileParams] = useState<PatientParams>({
    pregnancies: 0.3957,
    glucose: 0.7288,
    skin_thickness: 0.7968,
    bmi: 0.5861,
    age: 0.8721,
  })

  // Currently selected sample patient from dataset (if any)
  const [selectedSamplePatient, setSelectedSamplePatient] = useState<SamplePatient | null>(null)

  // Currently applied reference profile (if any)
  const [activePreset, setActivePreset] = useState<string | null>(null)

  // Prediction status & results
  const [predicting, setPredicting] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null)
  const [predictError, setPredictError] = useState<string | null>(null)

  // Selected views
  const [selectedModelName, setSelectedModelName] = useState<string>('Decision Tree')
  const [selectedPlotModal, setSelectedPlotModal] = useState<PlotItem | null>(null)
  const [plotFilter, setPlotFilter] = useState<'all' | 'eda' | 'performance' | 'shap'>('all')
  const [rankMetric, setRankMetric] = useState<RankMetric>('f1_score')

  // Predict function
  const runPrediction = async (params: PatientParams, isRaw: boolean) => {
    setPredicting(true)
    setPredictError(null)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...params,
          is_raw: isRaw,
        }),
      })

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}: ${res.statusText}`)
      }

      const data: PredictionResponse = await res.json()
      setPredictionResult(data)

      // Sync the other mode's parameters from the returned calculation
      if (isRaw && data.quantile_inputs) {
        setQuantileParams({
          pregnancies: data.quantile_inputs.pregnancies,
          glucose: data.quantile_inputs.glucose,
          skin_thickness: data.quantile_inputs.skin_thickness,
          bmi: data.quantile_inputs.bmi,
          age: data.quantile_inputs.age,
        })
      }
    } catch (err: any) {
      console.error('Prediction failed:', err)
      setPredictError(err.message || 'Failed to communicate with prediction service.')
    } finally {
      setPredicting(false)
    }
  }

  // Fetch every dataset/model resource the dashboard needs. Each request is
  // isolated so one failing endpoint never blanks out the rest of the UI.
  const loadAllData = async (showSpinner = true) => {
    if (showSpinner) setLoadingInitial(true)
    setLoadError(null)

    const healthRes = await fetch('/api/health').catch(() => null)
    const online = Boolean(healthRes && healthRes.ok)
    setBackendOnline(online)

    if (!online) {
      setLoadError('Cannot reach the prediction API on http://localhost:8000. Start the FastAPI backend, then retry.')
      setLoadingInitial(false)
      return
    }

    const fetchJson = async <T,>(url: string, fallback: T): Promise<T> => {
      try {
        const res = await fetch(url)
        if (!res.ok) return fallback
        return (await res.json()) as T
      } catch (err) {
        console.error(`Failed to fetch ${url}:`, err)
        return fallback
      }
    }

    const [mData, dsData, sData, pData] = await Promise.all([
      fetchJson<ModelMetric[]>('/api/models', []),
      fetchJson<DatasetInfo | null>('/api/dataset-info', null),
      fetchJson<SamplePatient[]>('/api/sample-patients', []),
      fetchJson<PlotItem[]>('/api/plots', []),
    ])

    setModels(mData)
    if (mData.length > 0) {
      const best = [...mData].sort((a, b) => b.accuracy - a.accuracy)[0]
      if (best) setSelectedModelName(best.model)
    }

    if (dsData) setDatasetInfo(dsData)

    setSamplePatients(sData)
    if (sData.length > 0) setSelectedSamplePatient(sData[0])

    setPlots(pData)

    // Seed the calculator with Patient #1 so the dashboard is never empty.
    await runPrediction(
      { pregnancies: 2, glucose: 138, skin_thickness: 35, bmi: 33.6, age: 47 },
      true,
    )

    setLoadingInitial(false)
  }

  const retryConnection = async () => {
    setRetrying(true)
    await loadAllData(false)
    setRetrying(false)
  }

  // Initial load
  useEffect(() => {
    loadAllData()
  }, [])

  if (loadingInitial) {
    return (
      <div style={{ display: 'flex', minHeight: '100vh', width: '100%', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-app)' }}>
        <div className="loading-container">
          <div className="spinner" />
          <div className="loading-text" style={{ fontSize: '15px', fontWeight: 500, color: 'var(--text-primary)' }}>
            Loading Dataset & Calculating ML Models...
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Processing <code>diabetes-dataset (1).csv</code> with QuantileTransformer, SMOTE & 5 Classifiers
          </div>
        </div>
      </div>
    )
  }

  // Handle parameter changes
  const handleClinicalChange = (field: keyof PatientParams, val: number) => {
    setSelectedSamplePatient(null)
    setActivePreset(null)
    const updated = { ...clinicalParams, [field]: val }
    setClinicalParams(updated)
  }

  const handleQuantileChange = (field: keyof PatientParams, val: number) => {
    setSelectedSamplePatient(null)
    setActivePreset(null)
    const updated = { ...quantileParams, [field]: val }
    setQuantileParams(updated)
  }

  // Apply a reference clinical profile and immediately score it
  const applyPreset = (preset: ClinicalPreset) => {
    setSelectedSamplePatient(null)
    setActivePreset(preset.label)
    setClinicalParams(preset.params)
    runPrediction(preset.params, true)
  }

  // Load a patient record from the dataset
  const selectDatasetPatient = (patient: SamplePatient) => {
    setSelectedSamplePatient(patient)
    setActivePreset(null)
    const pVals: PatientParams = {
      pregnancies: patient.pregnancies,
      glucose: patient.glucose,
      skin_thickness: patient.skin_thickness,
      bmi: patient.bmi,
      age: patient.age,
    }
    setClinicalParams(pVals)
    setInputMode('clinical')
    runPrediction(pVals, true)
  }

  // Ensemble statistics
  const consensusRisk = predictionResult
    ? (
        predictionResult.predictions.reduce((acc, curr) => acc + curr.probability_diabetes, 0) /
        predictionResult.predictions.length
      )
    : 0

  const consensusIsHigh = consensusRisk >= 0.5
  const predictedClass = consensusIsHigh ? 1 : 0

  // Check if consensus matches ground truth from dataset (if a sample patient was loaded)
  const groundTruthMatch = selectedSamplePatient
    ? predictedClass === selectedSamplePatient.actual_outcome
    : null

  // Best model
  const bestModel = models.length > 0
    ? [...models].sort((a, b) => b.accuracy - a.accuracy)[0]
    : null

  // Categorize plots
  const getPlotCategory = (filename: string) => {
    if (filename.includes('shap')) return 'shap'
    if (filename.includes('distribution') || filename.includes('histograms') || filename.includes('scatter') || filename.includes('heatmap') || filename.includes('boxplots')) return 'eda'
    return 'performance'
  }

  const filteredPlots = plots.filter(p => {
    if (plotFilter === 'all') return true
    return getPlotCategory(p.filename) === plotFilter
  })

  const selectedModelMetric = models.find(m => m.model === selectedModelName) || models[0]

  // --- Evaluation tab derivations --------------------------------
  const selectedMetrics = selectedModelMetric
    ? clinicalMetrics(selectedModelMetric.confusion_matrix)
    : null

  // Screening prioritises recall; ties break on F1 so a model that
  // flags everyone cannot win on recall alone.
  const screeningPick = models.length > 0
    ? [...models].sort((a, b) => (b.recall - a.recall) || (b.f1_score - a.f1_score))[0]
    : null

  const rankedModels = [...models].sort((a, b) => b[rankMetric] - a[rankMetric])
  const rankMetricLabel = RANK_METRIC_LABELS[rankMetric]

  const evaluationPlots = plots.filter(p =>
    ['roc', 'calibration', 'confusion', 'model_comparison'].some(k => p.filename.includes(k)),
  )

  return (
    <div className="app-layout">
      {/* Mobile Top Header */}
      <header className="mobile-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon"><Icon name="logo" size={17} /></div>
          <div className="sidebar-logo-text">
            <h1>DiaPredict AI</h1>
            <span>Clinical ML Platform</span>
          </div>
        </div>
        <button
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label={mobileMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={mobileMenuOpen}
        >
          <Icon name={mobileMenuOpen ? 'close' : 'menu'} size={18} />
        </button>
      </header>

      {mobileMenuOpen && (
        <div className="sidebar-backdrop" onClick={() => setMobileMenuOpen(false)} />
      )}

      {/* Navigation Sidebar */}
      <aside className={`sidebar ${mobileMenuOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon"><Icon name="logo" size={17} /></div>
            <div className="sidebar-logo-text">
              <h1>DiaPredict AI</h1>
              <span>Clinical ML Platform</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV_GROUPS.map(group => (
            <div key={group.title} className="nav-section">
              <div className="nav-section-title">{group.title}</div>
              {group.items.map(item => (
                <button
                  key={item.id}
                  type="button"
                  className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                  onClick={() => { setActiveTab(item.id); setMobileMenuOpen(false); }}
                  aria-current={activeTab === item.id ? 'page' : undefined}
                >
                  <span className="nav-item-icon"><Icon name={item.icon} /></span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        {/* Backend / dataset status */}
        <div className="sidebar-footer">
          <div className={`status-pill ${backendOnline ? '' : 'offline'}`}>
            <span className="status-dot" />
            <span>{backendOnline ? 'API Connected' : 'API Offline'}</span>
          </div>
          <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-faint)', lineHeight: 1.5 }}>
            {backendOnline && datasetInfo
              ? `${datasetInfo.shape_after_dedup[0]} patients · ${models.length} classifiers`
              : 'Waiting for FastAPI on port 8000'}
          </div>
          {!backendOnline && (
            <button
              className="filter-btn"
              style={{ marginTop: '10px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              onClick={retryConnection}
              disabled={retrying}
            >
              <Icon name="refresh" size={13} />
              {retrying ? 'Reconnecting…' : 'Retry Connection'}
            </button>
          )}
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {loadError && (
          <div
            className="ground-truth-match-banner mismatch"
            style={{ marginBottom: '24px', alignItems: 'center' }}
          >
            <span className="banner-icon"><Icon name="offline" size={18} /></span>
            <div style={{ flex: 1 }}>
              <div><strong>Backend Unavailable</strong></div>
              <div style={{ fontSize: '12px', marginTop: '2px' }}>{loadError}</div>
              <div style={{ fontSize: '12px', marginTop: '6px', color: 'var(--text-muted)' }}>
                Run <code>uvicorn main:app --reload</code> from the <code>backend/</code> directory.
              </div>
            </div>
            <button className="filter-btn" onClick={retryConnection} disabled={retrying}>
              {retrying ? 'Reconnecting…' : 'Retry'}
            </button>
          </div>
        )}

        {/* ============================================================
            TAB 1: RISK PREDICTOR & DATASET CALCULATIONS
            ============================================================ */}
        {activeTab === 'predict' && (
          <div>
            <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
              <div>
                <h2>Clinical Diabetes Risk Calculator</h2>
                <p>
                  Calculations performed using the trained QuantileTransformer and 5 ML classifiers fitted on <code>diabetes-dataset (1).csv</code>.
                </p>
              </div>

              {/* Mode Switcher: Clinical vs Quantile */}
              <div className="mode-toggle-bar">
                <button
                  className={`mode-toggle-btn ${inputMode === 'clinical' ? 'active' : ''}`}
                  onClick={() => setInputMode('clinical')}
                >
                  Clinical Units (mg/dL, kg/m²)
                </button>
                <button
                  className={`mode-toggle-btn ${inputMode === 'quantile' ? 'active' : ''}`}
                  onClick={() => setInputMode('quantile')}
                >
                  Quantile Scale [0.0 - 1.0]
                </button>
              </div>
            </div>

            {/* Dataset Patient Browser — Load actual patients from diabetes-dataset (1).csv */}
            <div className="dataset-browser-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'flex' }}><Icon name="patients" size={16} /></span>
                  <div>
                    <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                      Quick Load Patient Record from Dataset
                    </strong>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: '8px' }}>
                      (First {samplePatients.length} patients from <code>diabetes-dataset (1).csv</code>)
                    </span>
                  </div>
                </div>
                {selectedSamplePatient && (
                  <span style={{ fontSize: '12px', color: 'var(--accent-text)', fontWeight: '600' }}>
                    Active: Patient #{selectedSamplePatient.id}
                  </span>
                )}
              </div>

              {samplePatients.length === 0 && (
                <div style={{ fontSize: '13px', color: 'var(--text-muted)', padding: '8px 0' }}>
                  No patient records loaded — the dataset is served by the backend.
                </div>
              )}

              <div className="dataset-patient-chips">
                {samplePatients.map(sp => {
                  const isSelected = selectedSamplePatient?.id === sp.id
                  const isDiabetic = sp.actual_outcome === 1
                  return (
                    <div
                      key={sp.id}
                      className={`dataset-patient-chip ${isSelected ? 'active' : ''}`}
                      onClick={() => selectDatasetPatient(sp)}
                    >
                      <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-primary)' }}>
                        Patient #{sp.id}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        Gluc: <strong>{sp.glucose}</strong> | BMI: <strong>{sp.bmi}</strong> | Age: <strong>{sp.age}</strong>
                      </div>
                      <span className={`truth-badge ${isDiabetic ? 'diabetic' : 'healthy'}`}>
                        Actual: {isDiabetic ? 'Diabetic (1)' : 'Healthy (0)'}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="prediction-container">
              {/* Left Column: Form Controls */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <div className="card-title">
                      {inputMode === 'clinical' ? 'Patient Clinical Attributes (Real Units)' : 'Quantile Transformed Parameters [0, 1]'}
                    </div>
                    <div className="card-subtitle">
                      {inputMode === 'clinical'
                        ? 'Empirical values converted via QuantileTransformer fit on dataset'
                        : 'Normalized percentile inputs across empirical distribution'}
                    </div>
                  </div>
                </div>

                {inputMode === 'clinical' ? (
                  /* Clinical Real-World Inputs */
                  <>
                    {/* Reference risk profiles */}
                    <div className="presets-section">
                      <div className="presets-title">Reference Clinical Profiles</div>
                      <div className="presets-container">
                        {CLINICAL_PRESETS.map(preset => (
                          <button
                            key={preset.label}
                            className={`preset-chip ${activePreset === preset.label ? 'active' : ''}`}
                            title={preset.description}
                            onClick={() => applyPreset(preset)}
                            disabled={predicting}
                          >
                            <span className={`preset-dot ${preset.tier}`} />
                            {preset.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Pregnancies */}
                    <div className="form-group">
                      <div className="form-label">
                        <span>Pregnancies (Count)</span>
                        <span className="form-value">{clinicalParams.pregnancies}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="17"
                          step="1"
                          value={clinicalParams.pregnancies}
                          onChange={e => handleClinicalChange('pregnancies', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="param-hint">Dataset Range: 0 to 17 pregnancies (Median: 3)</div>
                    </div>

                    {/* Glucose */}
                    <div className="form-group">
                      <div className="form-label">
                        <span>Glucose (Plasma Concentration)</span>
                        <span className="form-value">{clinicalParams.glucose} mg/dL</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="44"
                          max="199"
                          step="1"
                          value={clinicalParams.glucose}
                          onChange={e => handleClinicalChange('glucose', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="param-hint">
                        Dataset Range: 44 to 199 mg/dL (Normal Fasting: 70-99 mg/dL, Mean: 120.9)
                      </div>
                    </div>

                    {/* Skin Thickness */}
                    <div className="form-group">
                      <div className="form-label">
                        <span>Skin Thickness (Triceps Skinfold)</span>
                        <span className="form-value">{clinicalParams.skin_thickness} mm</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="7"
                          max="99"
                          step="1"
                          value={clinicalParams.skin_thickness}
                          onChange={e => handleClinicalChange('skin_thickness', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="param-hint">Dataset Range: 7 to 99 mm (Median: 23 mm)</div>
                    </div>

                    {/* BMI */}
                    <div className="form-group">
                      <div className="form-label">
                        <span>Body Mass Index (BMI)</span>
                        <span className="form-value">{clinicalParams.bmi.toFixed(1)} kg/m²</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="18.2"
                          max="67.1"
                          step="0.1"
                          value={clinicalParams.bmi}
                          onChange={e => handleClinicalChange('bmi', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="param-hint">Dataset Range: 18.2 to 67.1 kg/m² (Normal: 18.5-24.9, Median: 32.3)</div>
                    </div>

                    {/* Age */}
                    <div className="form-group">
                      <div className="form-label">
                        <span>Age</span>
                        <span className="form-value">{clinicalParams.age} years</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="21"
                          max="81"
                          step="1"
                          value={clinicalParams.age}
                          onChange={e => handleClinicalChange('age', parseFloat(e.target.value))}
                        />
                      </div>
                      <div className="param-hint">Dataset Range: 21 to 81 years (Median: 29)</div>
                    </div>

                    <button
                      className="predict-button"
                      onClick={() => runPrediction(clinicalParams, true)}
                      disabled={predicting}
                    >
                      {predicting ? 'Calculating ML Predictions...' : 'Calculate Diabetes Risk (Transform & Predict)'}
                    </button>
                  </>
                ) : (
                  /* Quantile Scale Inputs [0, 1] */
                  <>
                    <div className="form-group">
                      <div className="form-label">
                        <span>Pregnancies Percentile</span>
                        <span className="form-value">{quantileParams.pregnancies.toFixed(4)}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={quantileParams.pregnancies}
                          onChange={e => handleQuantileChange('pregnancies', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>

                    <div className="form-group">
                      <div className="form-label">
                        <span>Glucose Percentile</span>
                        <span className="form-value">{quantileParams.glucose.toFixed(4)}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={quantileParams.glucose}
                          onChange={e => handleQuantileChange('glucose', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>

                    <div className="form-group">
                      <div className="form-label">
                        <span>Skin Thickness Percentile</span>
                        <span className="form-value">{quantileParams.skin_thickness.toFixed(4)}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={quantileParams.skin_thickness}
                          onChange={e => handleQuantileChange('skin_thickness', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>

                    <div className="form-group">
                      <div className="form-label">
                        <span>BMI Percentile</span>
                        <span className="form-value">{quantileParams.bmi.toFixed(4)}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={quantileParams.bmi}
                          onChange={e => handleQuantileChange('bmi', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>

                    <div className="form-group">
                      <div className="form-label">
                        <span>Age Percentile</span>
                        <span className="form-value">{quantileParams.age.toFixed(4)}</span>
                      </div>
                      <div className="slider-container">
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={quantileParams.age}
                          onChange={e => handleQuantileChange('age', parseFloat(e.target.value))}
                        />
                      </div>
                    </div>

                    <button
                      className="predict-button"
                      onClick={() => runPrediction(quantileParams, false)}
                      disabled={predicting}
                    >
                      {predicting ? 'Calculating ML Predictions...' : 'Calculate Risk with Quantile Scale'}
                    </button>
                  </>
                )}

                {predictError && (
                  <div style={{ marginTop: '16px', padding: '12px', background: 'var(--danger-soft)', border: '1px solid rgba(212, 73, 79, 0.3)', borderRadius: '8px', color: 'var(--danger-text)', fontSize: '13px' }}>
                    ⚠️ {predictError}
                  </div>
                )}
              </div>

              {/* Right Column: Risk Gauge & Ground Truth Verification */}
              <div className="card">
                <div className="card-header">
                  <div>
                    <div className="card-title">Ensemble Consensus Risk Assessment</div>
                    <div className="card-subtitle">Mean predicted probability across all 5 ML models</div>
                  </div>
                </div>

                {predicting ? (
                  <div className="loading-container">
                    <div className="spinner" />
                    <div className="loading-text">Computing ML inferences & SHAP values...</div>
                  </div>
                ) : predictionResult ? (
                  <div>
                    <div className="risk-gauge-container">
                      <div className="risk-gauge">
                        <svg viewBox="0 0 200 110">
                          <path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke="var(--bg-inset)"
                            strokeWidth="16"
                            strokeLinecap="round"
                          />
                          <path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke={consensusIsHigh ? 'var(--danger)' : 'var(--success)'}
                            strokeWidth="16"
                            strokeLinecap="round"
                            strokeDasharray="251.2"
                            strokeDashoffset={251.2 * (1 - consensusRisk)}
                            style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.4s ease' }}
                          />
                        </svg>
                      </div>

                      <div className="risk-percentage" style={{ color: consensusIsHigh ? 'var(--danger-text)' : 'var(--success-text)' }}>
                        {(consensusRisk * 100).toFixed(1)}%
                      </div>
                      <div className="stat-card-label" style={{ textAlign: 'center' }}>
                        Mean Predicted Probability of Diabetes
                      </div>
                      <div className={`risk-label ${consensusIsHigh ? 'high' : 'low'}`}>
                        <Icon name={consensusIsHigh ? 'alert' : 'check'} size={13} />
                        {consensusIsHigh ? 'High Risk Detected' : 'Low Risk Profile'}
                      </div>
                    </div>

                    {/* Dataset Ground Truth Verification Banner */}
                    {selectedSamplePatient && groundTruthMatch !== null && (
                      <div className={`ground-truth-match-banner ${groundTruthMatch ? 'success' : 'mismatch'}`}>
                        <span className="banner-icon"><Icon name={groundTruthMatch ? 'check' : 'alert'} size={16} /></span>
                        <div>
                          <div>
                            <strong>Dataset Validation Comparison:</strong>
                          </div>
                          <div style={{ fontSize: '12px', marginTop: '2px' }}>
                            Actual Outcome in Dataset: <strong>{selectedSamplePatient.actual_outcome_label}</strong> | Model Consensus: <strong>{consensusIsHigh ? 'Diabetic (High Risk)' : 'Non-Diabetic (Low Risk)'}</strong>
                            {' '}— {groundTruthMatch ? 'Prediction matches ground truth.' : 'Discrepancy noted.'}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Quick Model Vote Breakdown */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '16px' }}>
                      <div style={{ padding: '12px', background: 'var(--bg-inset)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>High Risk Votes</div>
                        <div style={{ fontSize: '20px', fontWeight: '700', marginTop: '4px', color: 'var(--danger-text)' }}>
                          {predictionResult.predictions.filter(p => p.prediction === 1).length} / {predictionResult.predictions.length}
                        </div>
                      </div>
                      <div style={{ padding: '12px', background: 'var(--bg-inset)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Low Risk Votes</div>
                        <div style={{ fontSize: '20px', fontWeight: '700', marginTop: '4px', color: 'var(--success-text)' }}>
                          {predictionResult.predictions.filter(p => p.prediction === 0).length} / {predictionResult.predictions.length}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <EmptyState
                    icon="calculator"
                    title="No Prediction Yet"
                    message="Adjust the patient attributes and run the calculator to see the ensemble risk assessment."
                  />
                )}
              </div>
            </div>

            {/* Calculations Transparency Pipeline */}
            {predictionResult && (
              <div className="card" style={{ marginTop: '24px' }}>
                <div className="card-header">
                  <div>
                    <div className="card-title">Mathematical & Transformation Pipeline Breakdown</div>
                    <div className="card-subtitle">
                      Step-by-step audit of how raw dataset measurements are transformed into predictions
                    </div>
                  </div>
                </div>

                <div className="calc-pipeline-container">
                  {/* Step 1 */}
                  <div className="calc-step-card">
                    <div className="calc-step-num">Step 1 • Ingestion</div>
                    <div className="calc-step-title">Raw Clinical Features</div>
                    <div className="calc-step-desc">
                      {predictionResult.raw_inputs ? (
                        <>
                          Preg: <strong>{predictionResult.raw_inputs.pregnancies}</strong><br />
                          Gluc: <strong>{predictionResult.raw_inputs.glucose} mg/dL</strong><br />
                          Skin: <strong>{predictionResult.raw_inputs.skin_thickness} mm</strong><br />
                          BMI: <strong>{predictionResult.raw_inputs.bmi} kg/m²</strong><br />
                          Age: <strong>{predictionResult.raw_inputs.age} yrs</strong>
                        </>
                      ) : (
                        'User passed direct quantile percentile values.'
                      )}
                    </div>
                  </div>

                  {/* Step 2 */}
                  <div className="calc-step-card">
                    <div className="calc-step-num">Step 2 • Quantile Transform</div>
                    <div className="calc-step-title">Uniform Mapping [0, 1]</div>
                    <div className="calc-step-desc">
                      Preg: <strong>{predictionResult.quantile_inputs.pregnancies.toFixed(3)}</strong><br />
                      Gluc: <strong>{predictionResult.quantile_inputs.glucose.toFixed(3)}</strong><br />
                      Skin: <strong>{predictionResult.quantile_inputs.skin_thickness.toFixed(3)}</strong><br />
                      BMI: <strong>{predictionResult.quantile_inputs.bmi.toFixed(3)}</strong><br />
                      Age: <strong>{predictionResult.quantile_inputs.age.toFixed(3)}</strong>
                    </div>
                  </div>

                  {/* Step 3 */}
                  <div className="calc-step-card">
                    <div className="calc-step-num">Step 3 • Standard Scaling</div>
                    <div className="calc-step-title">Distance Normalization</div>
                    <div className="calc-step-desc">
                      Applied standard z-score scaling: <code>(X - μ) / σ</code> for distance-sensitive models (KNN and Logistic Regression).
                    </div>
                  </div>

                  {/* Step 4 */}
                  <div className="calc-step-card">
                    <div className="calc-step-num">Step 4 • Ensemble Vote</div>
                    <div className="calc-step-title">5 ML Estimators</div>
                    <div className="calc-step-desc">
                      Consensus: <strong>{(consensusRisk * 100).toFixed(1)}%</strong><br />
                      Votes: <strong>{predictionResult.predictions.filter(p => p.prediction === 1).length} High / {predictionResult.predictions.filter(p => p.prediction === 0).length} Low</strong>
                    </div>
                  </div>

                  {/* Step 5 */}
                  <div className="calc-step-card">
                    <div className="calc-step-num">Step 5 • Explainability</div>
                    <div className="calc-step-title">TreeExplainer (SHAP)</div>
                    <div className="calc-step-desc">
                      Base Value: <strong>{predictionResult.base_value.toFixed(3)}</strong><br />
                      Dominant driver: <strong>{predictionResult.shap_explanation[0]?.feature} ({predictionResult.shap_explanation[0]?.shap_value > 0 ? '+' : ''}{predictionResult.shap_explanation[0]?.shap_value.toFixed(3)})</strong>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Individual Model Output Cards */}
            {predictionResult && (
              <div style={{ marginTop: '24px' }}>
                <div className="card-header" style={{ marginBottom: '12px' }}>
                  <div>
                    <div className="card-title">Model-by-Model Predictions & Probability Outputs</div>
                    <div className="card-subtitle">Exact posterior probability from each trained classifier</div>
                  </div>
                </div>

                <div className="results-grid">
                  {predictionResult.predictions.map(pred => {
                    const isHigh = pred.prediction === 1
                    return (
                      <div key={pred.model_name} className="result-card">
                        <div className="result-card-header">
                          <span className="result-card-model">{pred.model_name}</span>
                          <span className={`result-card-badge ${isHigh ? 'badge-high' : 'badge-low'}`}>
                            {pred.risk_level}
                          </span>
                        </div>

                        <div className="result-bar-container">
                          <div
                            className="result-bar"
                            style={{
                              width: `${(pred.probability_diabetes * 100).toFixed(0)}%`,
                              background: isHigh ? 'var(--danger)' : 'var(--success)',
                            }}
                          />
                        </div>

                        <div className="result-probability">
                          <span>Diabetes Prob: <strong>{(pred.probability_diabetes * 100).toFixed(1)}%</strong></span>
                          <span>Non-Diabetic: <strong>{(pred.probability_no_diabetes * 100).toFixed(1)}%</strong></span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* SHAP Feature Contribution Waterfall */}
            {predictionResult && predictionResult.shap_explanation.length > 0 && (
              <div className="card" style={{ marginTop: '32px' }}>
                <div className="card-header">
                  <div>
                    <div className="card-title">SHAP Feature Attribution (TreeExplainer)</div>
                    <div className="card-subtitle">
                      Quantitative impact of each biomarker on the Random Forest diabetes probability
                    </div>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    Baseline Value: <strong>{predictionResult.base_value.toFixed(3)}</strong>
                  </div>
                </div>

                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
                  Features extending to the right (<span style={{ color: 'var(--danger-text)', fontWeight: '600' }}>Red</span>) increase risk probability. Features extending to the left (<span style={{ color: 'var(--accent-text)', fontWeight: '600' }}>Blue</span>) reduce risk towards non-diabetic.
                </p>

                <div className="shap-container">
                  {predictionResult.shap_explanation.map(item => {
                    const isPositive = item.shap_value >= 0
                    const absVal = Math.abs(item.shap_value)
                    const widthPercent = Math.min(Math.max((absVal / 0.15) * 45, 6), 48)

                    return (
                      <div key={item.feature} className="shap-bar-row">
                        <div className="shap-feature-name">{item.feature}</div>
                        <div className="shap-bar-track">
                          <div className="shap-center-line" />
                          <div
                            className={`shap-bar-fill ${isPositive ? 'positive' : 'negative'}`}
                            style={{ width: `${widthPercent}%` }}
                          >
                            <span><span className="shap-bar-feature">{item.feature}: </span>{item.value.toFixed(2)}</span>
                          </div>
                        </div>
                        <div className={`shap-value ${isPositive ? 'positive' : 'negative'}`}>
                          {item.shap_value > 0 ? `+${item.shap_value.toFixed(4)}` : item.shap_value.toFixed(4)}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================
            TAB 2: DASHBOARD OVERVIEW
            ============================================================ */}
        {activeTab === 'overview' && (
          <div>
            <div className="page-header">
              <h2>Machine Learning Pipeline Overview</h2>
              <p>Calculations and benchmarks from <code>diabetes-dataset (1).csv</code></p>
            </div>

            <div className="stats-grid">
              <div className="stat-card blue">
                <div className="stat-card-icon"><Icon name="database" size={15} /></div>
                <div className="stat-card-value">
                  {datasetInfo ? datasetInfo.shape_after_dedup[0] : '744'}
                </div>
                <div className="stat-card-label">Deduplicated Patients (from 2,000 raw rows)</div>
              </div>

              <div className="stat-card purple">
                <div className="stat-card-icon"><Icon name="cpu" size={15} /></div>
                <div className="stat-card-value">{models.length || '5'}</div>
                <div className="stat-card-label">Trained ML Classifiers</div>
              </div>

              <div className="stat-card green">
                <div className="stat-card-icon"><Icon name="target" size={15} /></div>
                <div className="stat-card-value">
                  {bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}%` : '73.2%'}
                </div>
                <div className="stat-card-label">
                  Peak Test Accuracy ({bestModel ? bestModel.model : 'Logistic Regression'})
                </div>
              </div>

              <div className="stat-card amber">
                <div className="stat-card-icon"><Icon name="compare" size={15} /></div>
                <div className="stat-card-value">
                  {datasetInfo ? datasetInfo.smote_train_size : '786'}
                </div>
                <div className="stat-card-label">SMOTE Balanced Training Instances</div>
              </div>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-header">
                  <div className="card-title">Data Preprocessing & Transformation Methodology</div>
                </div>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '14px', color: 'var(--text-secondary)' }}>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--success-text)', display: 'flex', flexShrink: 0, marginTop: '2px' }}><Icon name="check" size={14} /></span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Missing Value Imputation:</strong> Biological zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI replaced by mean (Gaussian) or median (skewed).
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--success-text)', display: 'flex', flexShrink: 0, marginTop: '2px' }}><Icon name="check" size={14} /></span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Quantile Transformation:</strong> Fitted <code>QuantileTransformer</code> on the dataset maps non-linear empirical distributions smoothly into uniform <code>[0, 1]</code> space.
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--success-text)', display: 'flex', flexShrink: 0, marginTop: '2px' }}><Icon name="check" size={14} /></span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>SMOTE Oversampling:</strong> Synthetic Minority Over-sampling equalizes positive (diabetic) class balance on the training fold (from 595 to 786 instances).
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--success-text)', display: 'flex', flexShrink: 0, marginTop: '2px' }}><Icon name="check" size={14} /></span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Explainable AI (SHAP):</strong> TreeExplainer provides local game-theoretic attribution for every individual prediction.
                    </div>
                  </li>
                </ul>
              </div>

              <div className="card">
                <div className="card-header">
                  <div className="card-title">Model Benchmarks on Test Partition</div>
                </div>
                <div className="metric-bars">
                  {models.map(m => (
                    <div key={m.model} className="metric-bar-group">
                      <div className="metric-bar-label">
                        <span className="metric-bar-name">{m.model}</span>
                        <span className="metric-bar-value">Acc: {(m.accuracy * 100).toFixed(1)}% | F1: {(m.f1_score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="metric-bar-track">
                        <div
                          className="metric-bar-fill"
                          style={{
                            width: `${(m.accuracy * 100).toFixed(0)}%`,
                            background: m.accuracy > 0.72 ? 'var(--success)' : 'var(--accent)',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  {models.length === 0 && (
                    <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                      Benchmarks appear once the backend has trained the classifiers.
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Class balance & partition breakdown */}
            {datasetInfo && (
              <div className="grid-3" style={{ marginTop: '24px' }}>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Outcome Class Balance</div>
                      <div className="card-subtitle">Before SMOTE resampling</div>
                    </div>
                  </div>
                  <div className="metric-bars">
                    {([
                      ['Non-Diabetic (0)', datasetInfo.outcome_distribution.no_diabetes, 'var(--success)'],
                      ['Diabetic (1)', datasetInfo.outcome_distribution.diabetes, 'var(--warning)'],
                    ] as const).map(([label, count, gradient]) => {
                      const total =
                        datasetInfo.outcome_distribution.no_diabetes +
                        datasetInfo.outcome_distribution.diabetes
                      const pct = total > 0 ? (count / total) * 100 : 0
                      return (
                        <div key={label} className="metric-bar-group">
                          <div className="metric-bar-label">
                            <span className="metric-bar-name">{label}</span>
                            <span className="metric-bar-value">{count} ({pct.toFixed(1)}%)</span>
                          </div>
                          <div className="metric-bar-track">
                            <div className="metric-bar-fill" style={{ width: `${pct}%`, background: gradient }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Train / Test Partition</div>
                      <div className="card-subtitle">Stratified 80 / 20 split</div>
                    </div>
                  </div>
                  <div className="metric-bars">
                    <div className="metric-bar-group">
                      <div className="metric-bar-label">
                        <span className="metric-bar-name">Training Set</span>
                        <span className="metric-bar-value">{datasetInfo.train_size} rows</span>
                      </div>
                    </div>
                    <div className="metric-bar-group">
                      <div className="metric-bar-label">
                        <span className="metric-bar-name">After SMOTE</span>
                        <span className="metric-bar-value">{datasetInfo.smote_train_size} rows</span>
                      </div>
                    </div>
                    <div className="metric-bar-group">
                      <div className="metric-bar-label">
                        <span className="metric-bar-name">Hold-out Test Set</span>
                        <span className="metric-bar-value">{datasetInfo.test_size} rows</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Predictive Features</div>
                      <div className="card-subtitle">
                        Retained after dropping BloodPressure, Insulin & DiabetesPedigreeFunction
                      </div>
                    </div>
                  </div>
                  <div className="presets-container">
                    {datasetInfo.features_used.map(f => (
                      <span key={f} className="preset-chip active" style={{ cursor: 'default' }}>{f}</span>
                    ))}
                  </div>
                  <div style={{ marginTop: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
                    Target variable: <code>{datasetInfo.target}</code>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================
            TAB 3: MODEL COMPARISON
            ============================================================ */}
        {activeTab === 'models' && (
          <div>
            <div className="page-header">
              <h2>Model Benchmark & Evaluation</h2>
              <p>
                Performance metrics calculated on the 20% hold-out test set
                {datasetInfo ? ` (${datasetInfo.test_size} patient samples)` : ''}
              </p>
            </div>

            {models.length === 0 ? (
              <div className="card">
                <EmptyState
                  icon="cpu"
                  title="No Model Metrics Available"
                  message="Model benchmarks are computed by the backend at startup. Start the FastAPI server and retry the connection."
                />
              </div>
            ) : (
            <div className="card">
              <div className="card-header">
                <div className="card-title">Algorithm Performance Comparison</div>
              </div>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall (Sensitivity)</th>
                      <th>F1-Score</th>
                      <th>Specificity</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map(m => {
                      const isTop = bestModel?.model === m.model
                      const isSelected = selectedModelName === m.model
                      return (
                        <tr
                          key={m.model}
                          className={`${isTop ? 'best-model-row' : ''}`}
                          style={{ cursor: 'pointer', background: isSelected ? 'rgba(59, 130, 246, 0.08)' : undefined }}
                          onClick={() => setSelectedModelName(m.model)}
                        >
                          <td style={{ fontWeight: '600' }}>
                            {m.model}
                            {isTop && <span className="best-badge">Top Accuracy</span>}
                          </td>
                          <td>{(m.accuracy * 100).toFixed(2)}%</td>
                          <td>{(m.precision * 100).toFixed(2)}%</td>
                          <td>{(m.recall * 100).toFixed(2)}%</td>
                          <td>{(m.f1_score * 100).toFixed(2)}%</td>
                          <td>{(m.specificity * 100).toFixed(2)}%</td>
                          <td>
                            <button
                              className={`filter-btn ${isSelected ? 'active' : ''}`}
                              style={{ padding: '4px 10px', fontSize: '11px' }}
                              onClick={(e) => { e.stopPropagation(); setSelectedModelName(m.model); }}
                            >
                              Inspect CM
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            )}

            {selectedModelMetric && (
              <div className="grid-2" style={{ marginTop: '24px' }}>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Confusion Matrix: {selectedModelMetric.model}</div>
                      <div className="card-subtitle">
                        Test evaluation on {datasetInfo?.test_size ?? '—'} unseen patients
                      </div>
                    </div>
                  </div>

                  <div className="cm-container">
                    <div className="cm-grid">
                      <div className="cm-cell tn">
                        <div className="cm-value" style={{ color: 'var(--success-text)' }}>
                          {selectedModelMetric.confusion_matrix[0][0]}
                        </div>
                        <div className="cm-label">True Negative (Healthy)</div>
                      </div>
                      <div className="cm-cell fp">
                        <div className="cm-value" style={{ color: 'var(--danger-text)' }}>
                          {selectedModelMetric.confusion_matrix[0][1]}
                        </div>
                        <div className="cm-label">False Positive (False Alarm)</div>
                      </div>
                      <div className="cm-cell fn">
                        <div className="cm-value" style={{ color: 'var(--danger-text)' }}>
                          {selectedModelMetric.confusion_matrix[1][0]}
                        </div>
                        <div className="cm-label">False Negative (Missed Risk)</div>
                      </div>
                      <div className="cm-cell tp">
                        <div className="cm-value" style={{ color: 'var(--success-text)' }}>
                          {selectedModelMetric.confusion_matrix[1][1]}
                        </div>
                        <div className="cm-label">True Positive (Detected Diabetes)</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Hyperparameter Optimization Details</div>
                      <div className="card-subtitle">GridSearchCV tuned parameters</div>
                    </div>
                  </div>

                  <div style={{ background: 'var(--bg-inset)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)', marginBottom: '16px' }}>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                      Best Hyperparameters Found
                    </div>
                    {Object.keys(selectedModelMetric.best_params).length > 0 ? (
                      <pre style={{ fontSize: '12px', color: 'var(--accent-text)', overflowX: 'auto' }}>
                        {JSON.stringify(selectedModelMetric.best_params, null, 2)}
                      </pre>
                    ) : (
                      <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                        Default regularized configuration.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Per-class classification report from sklearn */}
            {selectedModelMetric && (
              <div className="card" style={{ marginTop: '24px' }}>
                <div className="card-header">
                  <div>
                    <div className="card-title">
                      Classification Report: {selectedModelMetric.model}
                    </div>
                    <div className="card-subtitle">
                      Per-class precision, recall and F1 as reported by <code>sklearn.metrics</code>
                    </div>
                  </div>
                </div>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Class / Average</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(selectedModelMetric.classification_report)
                        .filter(([, v]) => v && typeof v === 'object')
                        .map(([key, v]: [string, any]) => (
                          <tr key={key}>
                            <td style={{ fontWeight: '600' }}>{classLabel(key)}</td>
                            <td>{(v.precision * 100).toFixed(2)}%</td>
                            <td>{(v.recall * 100).toFixed(2)}%</td>
                            <td>{(v['f1-score'] * 100).toFixed(2)}%</td>
                            <td>{v.support}</td>
                          </tr>
                        ))}
                      {typeof selectedModelMetric.classification_report.accuracy === 'number' && (
                        <tr className="best-model-row">
                          <td style={{ fontWeight: '600' }}>Overall Accuracy</td>
                          <td colSpan={3} style={{ textAlign: 'center' }}>
                            {(selectedModelMetric.classification_report.accuracy * 100).toFixed(2)}%
                          </td>
                          <td>{datasetInfo?.test_size ?? '—'}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================
            TAB 4: MODEL EVALUATION & RESULTS
            ============================================================ */}
        {activeTab === 'evaluation' && (
          <div>
            <div className="page-header">
              <h2>Model Evaluation &amp; Results</h2>
              <p>
                Diagnostic performance of each classifier on the hold-out test set
                {datasetInfo ? ` (${datasetInfo.test_size} unseen patients)` : ''}, read in clinical terms.
              </p>
            </div>

            {models.length === 0 ? (
              <div className="card">
                <EmptyState
                  icon="evaluate"
                  title="No Evaluation Results"
                  message="Evaluation is derived from model metrics computed by the backend at startup. Start the FastAPI server and retry the connection."
                />
              </div>
            ) : (
              <>
                {/* Recommendation — screening favours recall over raw accuracy */}
                {screeningPick && (
                  <div className="eval-recommendation">
                    <div className="eval-recommendation-icon">
                      <Icon name="spark" size={17} />
                    </div>
                    <div>
                      <div className="eval-recommendation-title">
                        Recommended for screening: {screeningPick.model}
                      </div>
                      <div className="eval-recommendation-body">
                        It reaches the highest recall ({pct(screeningPick.recall)}), catching{' '}
                        {clinicalMetrics(screeningPick.confusion_matrix).tp} of{' '}
                        {clinicalMetrics(screeningPick.confusion_matrix).tp + clinicalMetrics(screeningPick.confusion_matrix).fn}{' '}
                        diabetic patients and missing only{' '}
                        {clinicalMetrics(screeningPick.confusion_matrix).fn}. In diabetes screening a false
                        negative (an undiagnosed patient sent home) carries far more cost than a false
                        positive, which a follow-up test resolves — so recall is weighted above overall
                        accuracy here.
                        {bestModel && bestModel.model !== screeningPick.model && (
                          <> By raw accuracy the leader is {bestModel.model} ({pct(bestModel.accuracy)}).</>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Selector */}
                <div className="filter-tabs">
                  {models.map(m => (
                    <button
                      key={m.model}
                      className={`filter-btn ${selectedModelName === m.model ? 'active' : ''}`}
                      onClick={() => setSelectedModelName(m.model)}
                    >
                      {m.model}
                    </button>
                  ))}
                </div>

                {/* Diagnostic metrics for the selected model */}
                {selectedModelMetric && selectedMetrics && (
                  <div className="card" style={{ marginBottom: '16px' }}>
                    <div className="card-header">
                      <div>
                        <div className="card-title">Diagnostic Performance: {selectedModelMetric.model}</div>
                        <div className="card-subtitle">
                          Derived from the confusion matrix across {selectedMetrics.total} test patients
                        </div>
                      </div>
                    </div>

                    <div className="eval-metric-grid">
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">Sensitivity</div>
                        <div className="eval-metric-value">{pct(selectedMetrics.sensitivity)}</div>
                        <div className="eval-metric-hint">
                          Of {selectedMetrics.tp + selectedMetrics.fn} diabetic patients, {selectedMetrics.tp} correctly flagged
                        </div>
                      </div>
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">Specificity</div>
                        <div className="eval-metric-value">{pct(selectedMetrics.specificity)}</div>
                        <div className="eval-metric-hint">
                          Of {selectedMetrics.tn + selectedMetrics.fp} healthy patients, {selectedMetrics.tn} correctly cleared
                        </div>
                      </div>
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">PPV (Precision)</div>
                        <div className="eval-metric-value">{pct(selectedMetrics.ppv)}</div>
                        <div className="eval-metric-hint">
                          Share of flagged patients who are genuinely diabetic
                        </div>
                      </div>
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">NPV</div>
                        <div className="eval-metric-value">{pct(selectedMetrics.npv)}</div>
                        <div className="eval-metric-hint">
                          Share of cleared patients who are genuinely healthy
                        </div>
                      </div>
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">Balanced Accuracy</div>
                        <div className="eval-metric-value">{pct(selectedMetrics.balancedAccuracy)}</div>
                        <div className="eval-metric-hint">
                          Mean of sensitivity and specificity — unaffected by class imbalance
                        </div>
                      </div>
                      <div className="eval-metric-card">
                        <div className="eval-metric-label">Matthews Corr.</div>
                        <div className="eval-metric-value">{selectedMetrics.mcc.toFixed(3)}</div>
                        <div className="eval-metric-hint">
                          Balanced single score from −1 to +1; 0 equals random guessing
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Ranked leaderboard */}
                <div className="grid-2" style={{ marginBottom: '16px' }}>
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Ranking by {rankMetricLabel}</div>
                        <div className="card-subtitle">All classifiers on the same hold-out partition</div>
                      </div>
                    </div>

                    <div className="filter-tabs" style={{ marginBottom: '12px' }}>
                      {(['f1_score', 'recall', 'accuracy', 'precision'] as const).map(key => (
                        <button
                          key={key}
                          className={`filter-btn ${rankMetric === key ? 'active' : ''}`}
                          onClick={() => setRankMetric(key)}
                        >
                          {RANK_METRIC_LABELS[key]}
                        </button>
                      ))}
                    </div>

                    <div>
                      {rankedModels.map((m, i) => (
                        <div key={m.model} className={`eval-rank-row ${i === 0 ? 'leader' : ''}`}>
                          <div className="eval-rank-num">{i + 1}</div>
                          <div className="eval-rank-body">
                            <div className="eval-rank-name">{m.model}</div>
                            <div className="metric-bar-track">
                              <div
                                className="metric-bar-fill"
                                style={{
                                  width: `${m[rankMetric] * 100}%`,
                                  background: i === 0 ? 'var(--success)' : 'var(--accent)',
                                }}
                              />
                            </div>
                          </div>
                          <div style={{ fontSize: '13px', fontWeight: 600, width: '58px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                            {pct(m[rankMetric])}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Error profile */}
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Error Profile</div>
                        <div className="card-subtitle">
                          How each model's mistakes split between missed cases and false alarms
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                      {models.map(m => {
                        const cm = clinicalMetrics(m.confusion_matrix)
                        const correct = cm.tp + cm.tn
                        return (
                          <div key={m.model}>
                            <div className="metric-bar-label">
                              <span className="metric-bar-name">{m.model}</span>
                              <span className="metric-bar-value">
                                {cm.fn} missed · {cm.fp} false alarms
                              </span>
                            </div>
                            <div className="eval-split-bar">
                              <div className="eval-split-correct" style={{ width: `${(correct / cm.total) * 100}%` }} />
                              <div className="eval-split-fn" style={{ width: `${(cm.fn / cm.total) * 100}%` }} />
                              <div className="eval-split-fp" style={{ width: `${(cm.fp / cm.total) * 100}%` }} />
                            </div>
                          </div>
                        )
                      })}
                    </div>

                    <div className="eval-legend">
                      <span className="eval-legend-item">
                        <span className="eval-legend-swatch" style={{ background: 'var(--success)' }} />
                        Correct
                      </span>
                      <span className="eval-legend-item">
                        <span className="eval-legend-swatch" style={{ background: 'var(--danger)' }} />
                        False negative (missed diabetic)
                      </span>
                      <span className="eval-legend-item">
                        <span className="eval-legend-swatch" style={{ background: 'var(--warning)' }} />
                        False positive (false alarm)
                      </span>
                    </div>
                  </div>
                </div>

                {/* Full diagnostic table */}
                <div className="card" style={{ marginBottom: '16px' }}>
                  <div className="card-header">
                    <div>
                      <div className="card-title">Full Diagnostic Matrix</div>
                      <div className="card-subtitle">
                        Every classifier scored on the measures that matter for screening
                      </div>
                    </div>
                  </div>
                  <div className="table-container">
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Sensitivity</th>
                          <th>Specificity</th>
                          <th>PPV</th>
                          <th>NPV</th>
                          <th>Balanced Acc.</th>
                          <th>MCC</th>
                          <th>Youden J</th>
                        </tr>
                      </thead>
                      <tbody>
                        {models.map(m => {
                          const cm = clinicalMetrics(m.confusion_matrix)
                          return (
                            <tr
                              key={m.model}
                              className={screeningPick?.model === m.model ? 'best-model-row' : ''}
                              style={{ cursor: 'pointer' }}
                              onClick={() => setSelectedModelName(m.model)}
                            >
                              <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                {m.model}
                                {screeningPick?.model === m.model && <span className="best-badge">Best Recall</span>}
                              </td>
                              <td>{pct(cm.sensitivity)}</td>
                              <td>{pct(cm.specificity)}</td>
                              <td>{pct(cm.ppv)}</td>
                              <td>{pct(cm.npv)}</td>
                              <td>{pct(cm.balancedAccuracy)}</td>
                              <td>{cm.mcc.toFixed(3)}</td>
                              <td>{cm.youden.toFixed(3)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Evaluation figures */}
                {evaluationPlots.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Evaluation Figures</div>
                        <div className="card-subtitle">
                          ROC, calibration and confusion-matrix diagnostics — select to enlarge
                        </div>
                      </div>
                    </div>
                    <div className="grid-plots">
                      {evaluationPlots.map(plot => (
                        <div
                          key={plot.filename}
                          className="plot-card"
                          onClick={() => setSelectedPlotModal(plot)}
                        >
                          <img src={plot.url} alt={plot.title} loading="lazy" />
                          <div className="plot-card-title">{plot.title}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ============================================================
            TAB 5: ANALYTICS & PLOTS GALLERY
            ============================================================ */}
        {activeTab === 'plots' && (
          <div>
            <div className="page-header">
              <h2>Diagnostic & Publication Visualizations</h2>
              <p>Generated plots from exploratory data analysis and model evaluation</p>
            </div>

            <div className="filter-tabs">
              <button
                className={`filter-btn ${plotFilter === 'all' ? 'active' : ''}`}
                onClick={() => setPlotFilter('all')}
              >
                All Figures ({plots.length})
              </button>
              <button
                className={`filter-btn ${plotFilter === 'eda' ? 'active' : ''}`}
                onClick={() => setPlotFilter('eda')}
              >
                EDA & Distributions
              </button>
              <button
                className={`filter-btn ${plotFilter === 'performance' ? 'active' : ''}`}
                onClick={() => setPlotFilter('performance')}
              >
                Model Performance & ROC
              </button>
              <button
                className={`filter-btn ${plotFilter === 'shap' ? 'active' : ''}`}
                onClick={() => setPlotFilter('shap')}
              >
                Explainability (SHAP)
              </button>
            </div>

            {filteredPlots.length === 0 ? (
              <div className="card">
                <EmptyState
                  icon="gallery"
                  title={plots.length === 0 ? 'No Figures Available' : 'No Figures in This Category'}
                  message={
                    plots.length === 0
                      ? 'Plots are served from the plots/ directory by the backend. Start the FastAPI server and retry the connection.'
                      : 'Try a different category filter to see the remaining figures.'
                  }
                />
              </div>
            ) : (
            <div className="grid-plots">
              {filteredPlots.map(plot => (
                <div
                  key={plot.filename}
                  className="plot-card"
                  onClick={() => setSelectedPlotModal(plot)}
                >
                  <img src={plot.url} alt={plot.title} loading="lazy" />
                  <div className="plot-card-title">{plot.title}</div>
                </div>
              ))}
            </div>
            )}
          </div>
        )}

        {/* ============================================================
            TAB 5: DATASET & PREPROCESSING EXPLORER
            ============================================================ */}
        {activeTab === 'dataset' && !datasetInfo && (
          <div>
            <div className="page-header">
              <h2>Dataset Architecture & Imputation Profiling</h2>
              <p>Based on <code>diabetes-dataset (1).csv</code></p>
            </div>
            <div className="card">
              <EmptyState
                icon="database"
                title="Dataset Not Loaded"
                message="Dataset statistics are computed by the backend when it starts. Start the FastAPI server and retry the connection."
              />
            </div>
          </div>
        )}

        {activeTab === 'dataset' && datasetInfo && (
          <div>
            <div className="page-header">
              <h2>Dataset Architecture & Imputation Profiling</h2>
              <p>Based on <code>diabetes-dataset (1).csv</code></p>
            </div>

            <div className="stats-grid">
              <div className="stat-card blue">
                <div className="stat-card-icon"><Icon name="inbox" size={15} /></div>
                <div className="stat-card-value">{datasetInfo.original_shape[0]}</div>
                <div className="stat-card-label">Raw Rows in File</div>
              </div>
              <div className="stat-card green">
                <div className="stat-card-icon"><Icon name="check" size={15} /></div>
                <div className="stat-card-value">{datasetInfo.shape_after_dedup[0]}</div>
                <div className="stat-card-label">Unique Patients (Deduplicated)</div>
              </div>
              <div className="stat-card purple">
                <div className="stat-card-icon"><Icon name="layers" size={15} /></div>
                <div className="stat-card-value">{datasetInfo.features_used.length}</div>
                <div className="stat-card-label">Selected Features</div>
              </div>
              <div className="stat-card amber">
                <div className="stat-card-icon"><Icon name="compare" size={15} /></div>
                <div className="stat-card-value">
                  {((datasetInfo.outcome_distribution.diabetes / datasetInfo.shape_after_dedup[0]) * 100).toFixed(1)}%
                </div>
                <div className="stat-card-label">Diabetic Class Ratio (Raw)</div>
              </div>
            </div>

            <div className="card" style={{ marginBottom: '24px' }}>
              <div className="card-header">
                <div>
                  <div className="card-title">Missing Zero Analysis & Replacement Strategy</div>
                  <div className="card-subtitle">Zero counts detected in biomarkers and how they were handled</div>
                </div>
              </div>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Biomarker</th>
                      <th>Zero Count (Missing)</th>
                      <th>Missing %</th>
                      <th>Calculated Replacement Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(datasetInfo.zero_counts).map(([metric, count]) => {
                      const pct = ((count / datasetInfo.original_shape[0]) * 100).toFixed(1)
                      const defVal = datasetInfo.imputation_defaults?.[metric]
                      return (
                        <tr key={metric}>
                          <td style={{ fontWeight: '600' }}>{metric}</td>
                          <td style={{ color: count > 100 ? 'var(--danger-text)' : 'var(--text-primary)' }}>{count}</td>
                          <td>{pct}%</td>
                          <td>
                            {defVal ? (
                              <span>
                                {metric === 'Glucose' || metric === 'BloodPressure' ? 'Mean' : 'Median'}: <strong>{defVal}</strong>
                              </span>
                            ) : (
                              'Calculated during preprocessing'
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Dataset Summary Statistics</div>
                  <div className="card-subtitle">Exploratory metrics from the clean dataset</div>
                </div>
              </div>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Mean</th>
                      <th>Std Dev</th>
                      <th>Min</th>
                      <th>Median (50%)</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(datasetInfo.summary_statistics).map(([feat, stats]) => (
                      <tr key={feat}>
                        <td style={{ fontWeight: '600' }}>{feat}</td>
                        <td>{stats.mean?.toFixed(2)}</td>
                        <td>{stats.std?.toFixed(2)}</td>
                        <td>{stats.min?.toFixed(1)}</td>
                        <td>{stats['50%']?.toFixed(1)}</td>
                        <td>{stats.max?.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Plot Fullscreen Lightbox Modal */}
      {selectedPlotModal && (
        <div className="modal-overlay" onClick={() => setSelectedPlotModal(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedPlotModal(null)}>✕</button>
            <img src={selectedPlotModal.url} alt={selectedPlotModal.title} />
            <div style={{ padding: '20px', borderTop: '1px solid var(--border)' }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700' }}>{selectedPlotModal.title}</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginTop: '4px' }}>
                Figure saved to <code>plots/{selectedPlotModal.filename}</code>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
