# Backend — DiaPredict AI API

FastAPI service that trains the models, serves predictions with SHAP explanations,
and exposes dataset/evaluation metadata.

See the [root README](../README.md) for the full architecture and API reference.

## Run

```bash
cd backend && ../venv/bin/python -m uvicorn main:app --reload --port 8000
```

First start trains all five models (several minutes) and writes a cache. Later starts
load the cache and are ready in seconds.

## Install

From the repository root:

```bash
./venv/bin/python -m pip install -r backend/requirements.txt
```

## Layout of `main.py`

The file is one module, ordered top to bottom:

| Section | Contents |
| --- | --- |
| App setup | FastAPI instance, CORS |
| Paths | Dataset resolution (`~/Downloads/` then repo root) |
| Model cache | `_cache_fingerprint()`, `_save_cache()`, `_load_cache()` |
| Pydantic models | `PatientInput`, `PredictionResult` |
| Training | `train_all_models()` — the whole pipeline |
| Startup | `_load_cache()` or `train_all_models()` |
| Endpoints | `/api/health`, `/api/dataset-info`, `/api/sample-patients`, `/api/models`, `/api/plots`, `/api/predict` |

Trained artefacts live in two module-level dicts:

- `models_store` — five fitted estimators, `StandardScaler`, the feature-only
  `QuantileTransformer`, the SHAP `TreeExplainer`, feature names, imputation defaults
- `data_store` — precomputed JSON payloads (model metrics, dataset info, sample patients)

## Environment variables

| Variable | Effect |
| --- | --- |
| `DIAPREDICT_FORCE_RETRAIN` | Any non-empty value bypasses the cache and retrains |

## Cache

`model_cache/trained_models.joblib` (~1.9 MB, gitignored) is invalidated automatically
when the dataset file changes (path/size/mtime), when scikit-learn or NumPy versions
change, or when `CACHE_VERSION` is bumped after a pipeline change.

To wipe it:

```bash
rm -rf backend/model_cache
```

## Adding a model

1. Fit it inside `train_all_models()`.
2. Add it to the `predictions` dict so its metrics are computed.
3. Store the fitted estimator in `models_store`.
4. Add it to `model_configs` in `predict()`, with a flag for whether it needs scaling.
5. Bump `CACHE_VERSION` so existing caches are invalidated.

The frontend needs no change — every view is driven by the `/api/models` array.
