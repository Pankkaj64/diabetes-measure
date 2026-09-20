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

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'predict' | 'models' | 'plots' | 'dataset'>('predict')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)

  // Data states from backend
  const [models, setModels] = useState<ModelMetric[]>([])
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [samplePatients, setSamplePatients] = useState<SamplePatient[]>([])
  const [plots, setPlots] = useState<PlotItem[]>([])
  const [loadingInitial, setLoadingInitial] = useState(true)

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

  // Prediction status & results
  const [predicting, setPredicting] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null)
  const [predictError, setPredictError] = useState<string | null>(null)

  // Selected views
  const [selectedModelName, setSelectedModelName] = useState<string>('Decision Tree')
  const [selectedPlotModal, setSelectedPlotModal] = useState<PlotItem | null>(null)
  const [plotFilter, setPlotFilter] = useState<'all' | 'eda' | 'performance' | 'shap'>('all')

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

  // Initial load
  useEffect(() => {
    async function loadData() {
      try {
        setLoadingInitial(true)
        // Check health
        const healthRes = await fetch('/api/health').catch(() => null)
        if (healthRes && healthRes.ok) {
          setBackendOnline(true)
        } else {
          setBackendOnline(false)
        }

        // Fetch models
        const modelsRes = await fetch('/api/models')
        if (modelsRes.ok) {
          const mData = await modelsRes.json()
          setModels(mData)
          if (mData.length > 0) {
            const best = [...mData].sort((a: ModelMetric, b: ModelMetric) => b.accuracy - a.accuracy)[0]
            if (best) setSelectedModelName(best.model)
          }
        }

        // Fetch dataset info
        const dsRes = await fetch('/api/dataset-info')
        if (dsRes.ok) {
          const dsData = await dsRes.json()
          setDatasetInfo(dsData)
        }

        // Fetch sample patients from dataset
        const sampleRes = await fetch('/api/sample-patients')
        if (sampleRes.ok) {
          const sData: SamplePatient[] = await sampleRes.json()
          setSamplePatients(sData)
          if (sData.length > 0) {
            setSelectedSamplePatient(sData[0])
          }
        }

        // Fetch plots
        const plotsRes = await fetch('/api/plots')
        if (plotsRes.ok) {
          const pData = await plotsRes.json()
          setPlots(pData)
        }

        // Execute initial prediction with Patient #1 from dataset
        await runPrediction({
          pregnancies: 2,
          glucose: 138,
          skin_thickness: 35,
          bmi: 33.6,
          age: 47,
        }, true)
      } catch (err) {
        console.error('Failed to load backend data:', err)
        setBackendOnline(false)
      } finally {
        setLoadingInitial(false)
      }
    }

    loadData()
  }, [])

  if (loadingInitial) {
    return (
      <div style={{ display: 'flex', minHeight: '100vh', width: '100%', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)' }}>
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
    const updated = { ...clinicalParams, [field]: val }
    setClinicalParams(updated)
  }

  const handleQuantileChange = (field: keyof PatientParams, val: number) => {
    setSelectedSamplePatient(null)
    const updated = { ...quantileParams, [field]: val }
    setQuantileParams(updated)
  }

  // Load a patient record from the dataset
  const selectDatasetPatient = (patient: SamplePatient) => {
    setSelectedSamplePatient(patient)
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

  return (
    <div className="app-layout">
      {/* Mobile Top Header */}
      <header className="mobile-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🩺</div>
          <div className="sidebar-logo-text">
            <h1>DiaPredict AI</h1>
            <span>Clinical ML Platform</span>
          </div>
        </div>
        <button
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? '✕' : '☰'}
        </button>
      </header>

      {mobileMenuOpen && (
        <div className="sidebar-backdrop" onClick={() => setMobileMenuOpen(false)} />
      )}

      {/* Navigation Sidebar */}
      <aside className={`sidebar ${mobileMenuOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">🩺</div>
            <div className="sidebar-logo-text">
              <h1>DiaPredict AI</h1>
              <span>Dataset ML Calculations</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div
            className={`nav-item ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => { setActiveTab('predict'); setMobileMenuOpen(false); }}
          >
            <span className="nav-item-icon">⚡</span>
            <span>Risk Calculator</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => { setActiveTab('overview'); setMobileMenuOpen(false); }}
          >
            <span className="nav-item-icon">📊</span>
            <span>Dashboard Overview</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => { setActiveTab('models'); setMobileMenuOpen(false); }}
          >
            <span className="nav-item-icon">🏆</span>
            <span>Model Comparison</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'plots' ? 'active' : ''}`}
            onClick={() => { setActiveTab('plots'); setMobileMenuOpen(false); }}
          >
            <span className="nav-item-icon">🖼️</span>
            <span>Analytics & Plots</span>
          </div>

          <div
            className={`nav-item ${activeTab === 'dataset' ? 'active' : ''}`}
            onClick={() => { setActiveTab('dataset'); setMobileMenuOpen(false); }}
          >
            <span className="nav-item-icon">📁</span>
            <span>Dataset & Pipeline</span>
          </div>
        </nav>

        {/* Sidebar Footer Status */}
        <div style={{ padding: '20px', borderTop: '1px solid var(--border-color)' }}>
          <div className={`status-pill ${backendOnline ? '' : 'offline'}`}>
            <span className="status-dot" />
            <span>Dataset: <code>diabetes-dataset (1).csv</code></span>
          </div>
          <div style={{ marginTop: '10px', fontSize: '11px', color: 'var(--text-muted)' }}>
            744 Deduplicated Patients • 5 Classifiers
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
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
                  <span style={{ fontSize: '18px' }}>📁</span>
                  <div>
                    <strong style={{ fontSize: '14px', color: 'var(--text-primary)' }}>
                      Quick Load Patient Record from Dataset
                    </strong>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginLeft: '8px' }}>
                      (First 20 patients from <code>/Users/pankkaj/Downloads/diabetes-dataset (1).csv</code>)
                    </span>
                  </div>
                </div>
                {selectedSamplePatient && (
                  <span style={{ fontSize: '12px', color: 'var(--accent-cyan)', fontWeight: '600' }}>
                    Active: Patient #{selectedSamplePatient.id}
                  </span>
                )}
              </div>

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
                  <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', color: 'var(--accent-red)', fontSize: '13px' }}>
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
                            stroke="rgba(255,255,255,0.08)"
                            strokeWidth="16"
                            strokeLinecap="round"
                          />
                          <path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke={consensusIsHigh ? '#ef4444' : '#10b981'}
                            strokeWidth="16"
                            strokeLinecap="round"
                            strokeDasharray="251.2"
                            strokeDashoffset={251.2 * (1 - consensusRisk)}
                            style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.4s ease' }}
                          />
                        </svg>
                      </div>

                      <div className="risk-percentage" style={{ color: consensusIsHigh ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                        {(consensusRisk * 100).toFixed(1)}%
                      </div>
                      <div className="stat-card-label" style={{ textAlign: 'center' }}>
                        Mean Predicted Probability of Diabetes
                      </div>
                      <div className={`risk-label ${consensusIsHigh ? 'high' : 'low'}`}>
                        {consensusIsHigh ? '🚨 High Risk Detected' : '✅ Low Risk Profile'}
                      </div>
                    </div>

                    {/* Dataset Ground Truth Verification Banner */}
                    {selectedSamplePatient && groundTruthMatch !== null && (
                      <div className={`ground-truth-match-banner ${groundTruthMatch ? 'success' : 'mismatch'}`}>
                        <span style={{ fontSize: '18px' }}>{groundTruthMatch ? '🎯' : '⚠️'}</span>
                        <div>
                          <div>
                            <strong>Dataset Validation Comparison:</strong>
                          </div>
                          <div style={{ fontSize: '12px', marginTop: '2px' }}>
                            Actual Outcome in Dataset: <strong>{selectedSamplePatient.actual_outcome_label}</strong> | Model Consensus: <strong>{consensusIsHigh ? 'Diabetic (High Risk)' : 'Non-Diabetic (Low Risk)'}</strong>
                            {' '}— {groundTruthMatch ? '✅ Model Prediction matches Ground Truth!' : 'Discrepancy noted.'}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Quick Model Vote Breakdown */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '16px' }}>
                      <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>High Risk Votes</div>
                        <div style={{ fontSize: '20px', fontWeight: '700', marginTop: '4px', color: 'var(--accent-red)' }}>
                          {predictionResult.predictions.filter(p => p.prediction === 1).length} / {predictionResult.predictions.length}
                        </div>
                      </div>
                      <div style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Low Risk Votes</div>
                        <div style={{ fontSize: '20px', fontWeight: '700', marginTop: '4px', color: 'var(--accent-green)' }}>
                          {predictionResult.predictions.filter(p => p.prediction === 0).length} / {predictionResult.predictions.length}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}
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
                              background: isHigh
                                ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                                : 'linear-gradient(90deg, #10b981, #06b6d4)',
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
                  Features extending to the right (<span style={{ color: 'var(--accent-red)', fontWeight: '600' }}>Red</span>) increase risk probability. Features extending to the left (<span style={{ color: 'var(--accent-blue)', fontWeight: '600' }}>Blue</span>) reduce risk towards non-diabetic.
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
                            <span>{item.feature}: {item.value.toFixed(2)}</span>
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
                <div className="stat-card-icon">🗃️</div>
                <div className="stat-card-value">
                  {datasetInfo ? datasetInfo.shape_after_dedup[0] : '744'}
                </div>
                <div className="stat-card-label">Deduplicated Patients (from 2,000 raw rows)</div>
              </div>

              <div className="stat-card purple">
                <div className="stat-card-icon">🤖</div>
                <div className="stat-card-value">{models.length || '5'}</div>
                <div className="stat-card-label">Trained ML Classifiers</div>
              </div>

              <div className="stat-card green">
                <div className="stat-card-icon">🎯</div>
                <div className="stat-card-value">
                  {bestModel ? `${(bestModel.accuracy * 100).toFixed(1)}%` : '73.2%'}
                </div>
                <div className="stat-card-label">
                  Peak Test Accuracy ({bestModel ? bestModel.model : 'Logistic Regression'})
                </div>
              </div>

              <div className="stat-card amber">
                <div className="stat-card-icon">⚖️</div>
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
                    <span style={{ color: 'var(--accent-blue)' }}>✔</span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Missing Value Imputation:</strong> Biological zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI replaced by mean (Gaussian) or median (skewed).
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--accent-blue)' }}>✔</span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>Quantile Transformation:</strong> Fitted <code>QuantileTransformer</code> on the dataset maps non-linear empirical distributions smoothly into uniform <code>[0, 1]</code> space.
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--accent-blue)' }}>✔</span>
                    <div>
                      <strong style={{ color: 'var(--text-primary)' }}>SMOTE Oversampling:</strong> Synthetic Minority Over-sampling equalizes positive (diabetic) class balance on the training fold (from 595 to 786 instances).
                    </div>
                  </li>
                  <li style={{ display: 'flex', gap: '10px' }}>
                    <span style={{ color: 'var(--accent-blue)' }}>✔</span>
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
                            background: m.accuracy > 0.72 ? 'var(--gradient-primary)' : 'var(--gradient-accent)',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================
            TAB 3: MODEL COMPARISON
            ============================================================ */}
        {activeTab === 'models' && (
          <div>
            <div className="page-header">
              <h2>Model Benchmark & Evaluation</h2>
              <p>Performance metrics calculated on the 20% hold-out test set (149 patient samples)</p>
            </div>

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

            {selectedModelMetric && (
              <div className="grid-2" style={{ marginTop: '24px' }}>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Confusion Matrix: {selectedModelMetric.model}</div>
                      <div className="card-subtitle">Test evaluation on 149 unseen patients</div>
                    </div>
                  </div>

                  <div className="cm-container">
                    <div className="cm-grid">
                      <div className="cm-cell tn">
                        <div className="cm-value" style={{ color: 'var(--accent-green)' }}>
                          {selectedModelMetric.confusion_matrix[0][0]}
                        </div>
                        <div className="cm-label">True Negative (Healthy)</div>
                      </div>
                      <div className="cm-cell fp">
                        <div className="cm-value" style={{ color: 'var(--accent-red)' }}>
                          {selectedModelMetric.confusion_matrix[0][1]}
                        </div>
                        <div className="cm-label">False Positive (False Alarm)</div>
                      </div>
                      <div className="cm-cell fn">
                        <div className="cm-value" style={{ color: 'var(--accent-red)' }}>
                          {selectedModelMetric.confusion_matrix[1][0]}
                        </div>
                        <div className="cm-label">False Negative (Missed Risk)</div>
                      </div>
                      <div className="cm-cell tp">
                        <div className="cm-value" style={{ color: 'var(--accent-green)' }}>
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

                  <div style={{ background: 'rgba(0,0,0,0.2)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', marginBottom: '16px' }}>
                    <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                      Best Hyperparameters Found
                    </div>
                    {Object.keys(selectedModelMetric.best_params).length > 0 ? (
                      <pre style={{ fontSize: '12px', color: 'var(--accent-cyan)', overflowX: 'auto' }}>
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
          </div>
        )}

        {/* ============================================================
            TAB 4: ANALYTICS & PLOTS GALLERY
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
          </div>
        )}

        {/* ============================================================
            TAB 5: DATASET & PREPROCESSING EXPLORER
            ============================================================ */}
        {activeTab === 'dataset' && datasetInfo && (
          <div>
            <div className="page-header">
              <h2>Dataset Architecture & Imputation Profiling</h2>
              <p>Based on <code>/Users/pankkaj/Downloads/diabetes-dataset (1).csv</code></p>
            </div>

            <div className="stats-grid">
              <div className="stat-card blue">
                <div className="stat-card-icon">📥</div>
                <div className="stat-card-value">{datasetInfo.original_shape[0]}</div>
                <div className="stat-card-label">Raw Rows in File</div>
              </div>
              <div className="stat-card green">
                <div className="stat-card-icon">🧹</div>
                <div className="stat-card-value">{datasetInfo.shape_after_dedup[0]}</div>
                <div className="stat-card-label">Unique Patients (Deduplicated)</div>
              </div>
              <div className="stat-card purple">
                <div className="stat-card-icon">🔬</div>
                <div className="stat-card-value">{datasetInfo.features_used.length}</div>
                <div className="stat-card-label">Selected Features</div>
              </div>
              <div className="stat-card amber">
                <div className="stat-card-icon">⚖️</div>
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
                          <td style={{ color: count > 100 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{count}</td>
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
            <div style={{ padding: '20px', borderTop: '1px solid var(--border-color)' }}>
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
