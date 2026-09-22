# DiaPredict AI — Clinical Diabetes Risk Platform

A full-stack machine learning dashboard that predicts Type 2 diabetes risk from five
clinical measurements, explains every prediction with SHAP, and reports model
performance in clinical terms rather than raw accuracy.

The backend trains five classifiers on startup (then caches them), and the React
frontend turns those results into a risk calculator, a model benchmark suite and a
diagnostic evaluation view.

> **Not a medical device.** This is an educational/research project built on a public
> dataset. Nothing here is validated for clinical use and it must not inform real
> patient care.

---

## Table of Contents

- [Screens](#screens)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [The ML Pipeline](#the-ml-pipeline)
- [Model Results](#model-results)
- [API Reference](#api-reference)
- [Model Caching](#model-caching)
- [Project Structure](#project-structure)
- [Frontend Design System](#frontend-design-system)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Screens

The app is organised into six views, grouped by workflow stage:

| Group | View | What it does |
| --- | --- | --- |
| **Overview** | Dashboard | Pipeline summary, preprocessing methodology, class balance, train/test split |
| **Overview** | Risk Calculator | Score a patient from clinical values or quantile percentiles; per-model probabilities + SHAP attribution |
| **Model Performance** | Model Comparison | Benchmark table, confusion matrix, tuned hyperparameters, per-class classification report |
| **Model Performance** | Evaluation & Results | Screening recommendation, sensitivity/specificity/PPV/NPV/MCC, ranked leaderboard, error profile |
| **Data & Analysis** | Analytics & Plots | 12 EDA and evaluation figures with a lightbox |
| **Data & Analysis** | Dataset & Pipeline | Row counts, missing-value analysis, imputation strategy, summary statistics |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Browser (React SPA)                         │
│                                                                      │
│   App.tsx — single component tree, no state library                  │
│   ├── Sidebar (NAV_GROUPS config)                                    │
│   ├── Risk Calculator ──┐                                            │
│   ├── Dashboard         │  all views read from the same              │
│   ├── Model Comparison  │  fetched-once state                        │
│   ├── Evaluation ───────┤  (models, datasetInfo, plots,              │
│   ├── Plots Gallery     │   samplePatients, predictionResult)        │
│   └── Dataset Explorer ─┘                                            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  fetch('/api/…')
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Vite dev server (port 5173) — proxies /api              │
│              target: VITE_API_TARGET or http://localhost:8000        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI backend (port 8000)                     │
│                                                                      │
│   startup ──► _load_cache()  ──hit──►  models_store / data_store     │
│                    │                                                 │
│                   miss                                               │
│                    ▼                                                 │
│              train_all_models()  ──►  _save_cache()                  │
│                                                                      │
│   models_store: 5 estimators + scaler + quantile transformer         │
│                 + SHAP TreeExplainer + imputation defaults           │
│   data_store:   model metrics, dataset info, sample patients         │
│                                                                      │
│   GET  /api/health  /api/models  /api/dataset-info                   │
│        /api/sample-patients  /api/plots  /api/plots/{file}           │
│   POST /api/predict                                                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   diabetes-dataset (1).csv          backend/model_cache/
   (2,000 rows, read at startup)     trained_models.joblib
                                     (fingerprinted, ~1.9 MB)
```

### Design decisions

**Training happens in-process, not offline.** The backend owns the whole pipeline, so
the dataset is the single source of truth — there is no separate training script whose
output could drift from what the API serves. The cost is a slow first boot, which the
cache solves.

**All state is fetched once and derived.** The frontend holds no state library. Each
view derives what it needs from the same fetched objects, so the confusion matrix on
the Comparison tab and the sensitivity figure on the Evaluation tab can never disagree
— they are computed from the same array.

**Evaluation metrics are derived client-side.** Sensitivity, specificity, PPV, NPV,
balanced accuracy, MCC and Youden's J all come from the 2×2 confusion matrix the API
already returns, so the Evaluation view needs no extra endpoint.

**Per-endpoint failure isolation.** The loader fetches the four data endpoints in
parallel and each falls back independently, so one failing endpoint degrades a single
view instead of blanking the dashboard.

---

## Tech Stack

**Backend**
- FastAPI + Uvicorn
- scikit-learn (5 classifiers, GridSearchCV, QuantileTransformer, StandardScaler)
- imbalanced-learn (SMOTE)
- SHAP (TreeExplainer)
- pandas / NumPy
- joblib (model cache)

**Frontend**
- React 19 + TypeScript
- Vite 8
- Zero UI dependencies — all components, charts, gauges and icons are hand-built
  (inline SVG + CSS), so there is no component-library lock-in and no runtime cost
  beyond React itself

---

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | 3.11+ | 3.14 used in development |
| Node.js | **20.19+ or 22.12+** | Vite 8 requires it; Node 18 fails the build |
| npm | 9+ | |

The dataset file `diabetes-dataset (1).csv` must be present in the repository root
(it is committed) or in `~/Downloads/`.

---

## Quick Start

### 1. Backend

```bash
python3 -m venv venv
```

```bash
./venv/bin/python -m pip install -r backend/requirements.txt
```

```bash
cd backend && ../venv/bin/python -m uvicorn main:app --reload --port 8000
```

**The first start takes several minutes.** It runs a full `GridSearchCV` sweep across
all five models. Watch for:

```
🔄 Loading dataset and training models...
  Training KNN...
  ...
✅ All models trained and ready!
💾 Model cache written to .../backend/model_cache/trained_models.joblib
```

Every later start loads the cache instead and is ready in a few seconds:

```
⚡ Loaded trained models from cache (trained_models.joblib) — skipped retraining.
```

### 2. Frontend

In a second terminal:

```bash
cd frontend && npm install
```

```bash
cd frontend && npm run dev
```

Open **http://localhost:5173**.

### 3. Verify

```bash
curl -s http://localhost:8000/api/health
```

```json
{ "status": "ok", "models_loaded": true, "cache": { "enabled": true, "present": true } }
```

---

## Configuration

| Variable | Where | Default | Purpose |
| --- | --- | --- | --- |
| `VITE_API_TARGET` | frontend dev server | `http://localhost:8000` | Backend origin the Vite proxy forwards `/api` to. Set it when port 8000 is taken. |
| `DIAPREDICT_FORCE_RETRAIN` | backend | unset | Any non-empty value skips the cache and retrains from scratch. |

Running the backend on a different port:

```bash
cd backend && ../venv/bin/python -m uvicorn main:app --port 8001
```

```bash
cd frontend && VITE_API_TARGET=http://localhost:8001 npm run dev
```

---

## The ML Pipeline

Implemented in `train_all_models()` in [`backend/main.py`](backend/main.py).

### 1. Load and deduplicate
2,000 raw rows → **744 unique patients**. The source file contains a large number of
duplicated records; leaving them in would leak identical rows across the train/test
split and inflate scores.

### 2. Impute biological zeros
A zero glucose or BMI is physiologically impossible — these are encoded missing values:

| Biomarker | Zeros | % of rows | Replaced with |
| --- | --- | --- | --- |
| Insulin | 956 | 47.8% | Median (36.0) |
| SkinThickness | 573 | 28.6% | Median (23.0) |
| BloodPressure | 90 | 4.5% | Mean (72.11) |
| BMI | 28 | 1.4% | Median (32.3) |
| Glucose | 13 | 0.7% | Mean (121.7) |

Mean for roughly symmetric distributions, median for skewed ones.

### 3. Feature selection
`BloodPressure`, `Insulin` and `DiabetesPedigreeFunction` are dropped. Five predictors
remain: **Pregnancies, Glucose, SkinThickness, BMI, Age**.

### 4. Quantile transformation
`QuantileTransformer` maps each feature onto a uniform `[0, 1]` distribution. This
handles the heavy skew in Insulin/SkinThickness and makes the distance-based models
behave. A second transformer is fitted on the five predictors alone so the API can
convert raw clinical input at prediction time.

### 5. Stratified split
80/20 → **595 train / 149 test**, stratified on the outcome.

### 6. SMOTE
Classes are imbalanced (491 non-diabetic / 253 diabetic). SMOTE is applied to the
**training fold only** — 595 → **786 rows**. Applying it before the split would leak
synthetic neighbours of test rows into training.

### 7. Scaling
`StandardScaler` is applied for KNN and Logistic Regression (distance/gradient
sensitive). The tree-based models and Naive Bayes use the unscaled quantile features.

### 8. Training
`GridSearchCV` with `RepeatedStratifiedKFold` for KNN and Random Forest; plain
K-fold for the Decision Tree. This sweep is what makes the first boot slow.

### 9. Explainability
A SHAP `TreeExplainer` is built on the tuned Random Forest and returns per-feature
attributions for every individual prediction.

---

## Model Results

Hold-out test set, 149 unseen patients:

| Model | Accuracy | Precision | Recall | F1 | Specificity |
| --- | --- | --- | --- | --- | --- |
| KNN | 65.10% | 49.37% | 76.47% | 60.00% | 59.18% |
| Decision Tree | 72.48% | 57.35% | 76.47% | 65.55% | 70.41% |
| Random Forest | 69.13% | 54.55% | 58.82% | 56.60% | 74.49% |
| **Logistic Regression** | **73.15%** | 57.53% | **82.35%** | **67.74%** | 68.37% |
| Naive Bayes | 72.48% | 57.35% | 76.47% | 65.55% | 70.41% |

Tuned hyperparameters:

```
KNN            metric=euclidean, n_neighbors=17, p=1, weights=distance
Decision Tree  criterion=gini, max_depth=5, min_samples_leaf=20
Random Forest  max_features=sqrt, n_estimators=500
```

### Reading these numbers

The Evaluation view ranks by **recall**, not accuracy. In screening, a false negative
is an undiagnosed patient sent home; a false positive is a follow-up test. Those costs
are not symmetric. Logistic Regression catches 42 of 51 diabetic patients (missing 9)
while Random Forest — despite better specificity — misses 21.

Accuracy alone is also misleading at 66/34 class balance, which is why the Evaluation
view reports balanced accuracy and Matthews correlation.

---

## API Reference

Base URL `http://localhost:8000`. CORS is open to all origins in development.

### `GET /api/health`
Liveness plus cache state.
```json
{ "status": "ok", "models_loaded": true,
  "cache": { "enabled": true, "present": true, "path": "…/trained_models.joblib" } }
```

### `GET /api/dataset-info`
Row counts, zero/missing counts, imputation defaults, feature list, split sizes,
class distribution and `describe()` summary statistics.

### `GET /api/sample-patients`
First 20 real patient records with their true outcome — used by the calculator's
quick-load strip so predictions can be checked against ground truth.

### `GET /api/models`
Per-model accuracy, precision, recall, F1, specificity, confusion matrix, tuned
parameters and full sklearn classification report.

> Class keys in `classification_report` are floats (`"0.0"` / `"1.0"`), because the
> quantile-transformed target is float-typed. The frontend normalises them.

### `GET /api/plots` · `GET /api/plots/{filename}`
Lists the 12 figures in `plots/` with display titles; the second serves the PNG.

### `POST /api/predict`

```json
{ "pregnancies": 2, "glucose": 138, "skin_thickness": 35,
  "bmi": 33.6, "age": 47, "is_raw": true }
```

`is_raw: true` means clinical units (mg/dL, kg/m², years) and the server applies the
quantile transform. `false` means values are already `[0, 1]` percentiles.

Returns every model's probability, the transformed inputs, and SHAP attributions
sorted by absolute impact:

```json
{
  "is_raw": true,
  "raw_inputs": { "glucose": 138.0, "bmi": 33.6, "age": 47.0, "...": "" },
  "quantile_inputs": { "glucose": 0.7288, "bmi": 0.5861, "age": 0.8721, "...": "" },
  "predictions": [
    { "model_name": "KNN", "prediction": 1,
      "probability_diabetes": 0.9343, "risk_level": "HIGH RISK" }
  ],
  "shap_explanation": [ { "feature": "Age", "value": 0.8721, "shap_value": 0.1276 } ],
  "base_value": 0.5004
}
```

---

## Model Caching

A full training run takes several minutes. Fitted estimators are persisted to
`backend/model_cache/trained_models.joblib` (~1.9 MB, gitignored), cutting restarts
from **minutes to seconds**.

### Invalidation

The cache stores a fingerprint and retrains automatically when any part changes:

| Fingerprint field | Retrains when |
| --- | --- |
| `dataset` (path, size, mtime) | The CSV is edited or replaced |
| `sklearn`, `numpy` versions | Pickled estimators are not portable across versions |
| `CACHE_VERSION` | Bumped by hand after a pipeline change |

The SHAP explainer is **rebuilt on load rather than pickled** — it is cheap to
reconstruct from the restored Random Forest and avoids a brittle cross-version pickle.

Cache failures never take the API down: an unreadable or incomplete cache logs a
warning and falls back to training.

### Forcing a retrain

```bash
DIAPREDICT_FORCE_RETRAIN=1 ../venv/bin/python -m uvicorn main:app --port 8000
```

Or delete the cache:

```bash
rm -rf backend/model_cache
```

---

## Project Structure

```
.
├── backend/
│   ├── main.py              FastAPI app: pipeline, cache, endpoints
│   ├── requirements.txt
│   └── model_cache/         Generated, gitignored
├── frontend/
│   ├── src/
│   │   ├── App.tsx          All views, nav config, icons, derived metrics
│   │   ├── index.css        Design system + components + responsive
│   │   └── main.tsx
│   ├── vite.config.ts       /api proxy (VITE_API_TARGET)
│   └── package.json
├── plots/                   12 pre-generated PNG figures
├── diabetes-dataset (1).csv 2,000 rows, 9 columns
└── README.md
```

---

## Frontend Design System

Tokens live at the top of [`frontend/src/index.css`](frontend/src/index.css).

- **Neutral slate surfaces with a single accent blue.** Semantic red/green/amber are
  reserved for *meaning* — risk level, outcome class, connection status — never
  decoration. A stat card carries its colour in a 2px top rule and its icon; the
  number stays neutral so values read as data.
- **Inline SVG icons** bound to `currentColor`, so each icon inherits its own state
  colour.
- **Tabular numerals** on every metric so columns align.
- **Responsive at 1024 / 768 / 480 px.** The sidebar becomes an off-canvas drawer;
  wide tables scroll internally rather than forcing horizontal page scroll. Also
  handles `prefers-reduced-motion` and `hover: none` (so hover styles do not stick
  after a tap on touch devices).

---

## Development

```bash
cd frontend && npm run dev          # dev server with HMR
```

```bash
cd frontend && npm run build        # typecheck + production build
```

```bash
cd frontend && npm run lint         # oxlint
```

The backend runs with `--reload` for auto-restart. Note that a reload re-runs startup
— fast, because of the cache.

---

## Troubleshooting

**`npm run build` fails with `does not provide an export named 'styleText'`**
Node is too old. Vite 8 needs Node 20.19+ / 22.12+.
```bash
nvm use 22
```

**Frontend shows "Backend Unavailable"**
The API is not reachable at the proxy target. Confirm it directly:
```bash
curl -s http://localhost:8000/api/health
```
If something else owns port 8000, run the backend elsewhere and point the proxy at it
with `VITE_API_TARGET` (see [Configuration](#configuration)). The dashboard stays
usable while offline and has a **Retry Connection** button — no reload needed.

**`ImportError: No module named 'numpy._core._multiarray_umath'`, or
`library load disallowed by system policy` on macOS**
The virtualenv has broken or quarantined native extensions, usually because it was
moved or copied. Reinstall them fresh:
```bash
./venv/bin/python -m pip install --force-reinstall --no-cache-dir -r backend/requirements.txt
```
If `./venv/bin/pip` itself fails with `bad interpreter`, the venv was created under a
different path — use `./venv/bin/python -m pip` as above, or recreate the venv.

**Backend startup is slow every time**
The cache is not being written or is being invalidated. Check the startup log for
`💾 Model cache written` or a `♻️`/`⚠️` line explaining why it retrained.

**`n_quantiles (1000) is greater than the total number of samples (744)`**
Harmless sklearn warning — `n_quantiles` is clamped to the sample count.

---

## Dataset

A public Pima-Indians-style diabetes dataset: 2,000 rows × 9 columns, reducing to
**744 unique records** after deduplication. All fields are de-identified clinical
measurements — there is no personally identifying information.

| Column | Description |
| --- | --- |
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Family-history likelihood score |
| Age | Years |
| Outcome | 1 = diabetic, 0 = non-diabetic |
