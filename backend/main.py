# -*- coding: utf-8 -*-
"""
Diabetes Prediction Dashboard — FastAPI Backend
Serves ML predictions, model metrics, SHAP explanations, and analysis plots.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)

import shap

# ============================================================
# App Setup
# ============================================================
app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE_DIR / "plots"

# Primary dataset in Downloads folder
DATASET_PATH = Path.home() / "Downloads" / "diabetes-dataset (1).csv"
if not DATASET_PATH.exists():
    DATASET_PATH = BASE_DIR / "diabetes-dataset (1).csv"


# ============================================================
# Global State — populated on startup
# ============================================================
models_store = {}
data_store = {}


# ============================================================
# Pydantic Models
# ============================================================
class PatientInput(BaseModel):
    pregnancies: float
    glucose: float
    skin_thickness: float
    bmi: float
    age: float
    is_raw: bool = True


class PredictionResult(BaseModel):
    model_name: str
    prediction: int
    probability_no_diabetes: float
    probability_diabetes: float
    risk_level: str


# ============================================================
# Training Pipeline (runs once on startup)
# ============================================================
def train_all_models():
    """Train all models and store them for prediction."""
    print("🔄 Loading dataset and training models...")

    df = pd.read_csv(str(DATASET_PATH))
    original_shape = df.shape

    # Store dataset info
    zero_counts = {
        "BloodPressure": int(df[df['BloodPressure'] == 0].shape[0]),
        "Glucose": int(df[df['Glucose'] == 0].shape[0]),
        "SkinThickness": int(df[df['SkinThickness'] == 0].shape[0]),
        "Insulin": int(df[df['Insulin'] == 0].shape[0]),
        "BMI": int(df[df['BMI'] == 0].shape[0]),
    }

    # Deduplication
    df = df.drop_duplicates()

    # Summary stats before imputation
    summary_stats = json.loads(df.describe().to_json())

    # Imputation
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
    df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

    # Feature selection & transformation
    df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')

    # Fit dedicated QuantileTransformer for the 5 predictive features
    feature_cols = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age']
    feature_quantile = QuantileTransformer(random_state=42)
    X_raw_features = df_selected[feature_cols].copy()
    X_features_trans = feature_quantile.fit_transform(X_raw_features)

    quantile = QuantileTransformer(random_state=42)
    X_trans = quantile.fit_transform(df_selected)
    df_new = pd.DataFrame(X_trans, columns=['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome'])

    # Train/test split
    target_name = 'Outcome'
    y = df_new[target_name]
    X = df_new.drop(target_name, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # ---- KNN ----
    print("  Training KNN...")
    knn = KNeighborsClassifier()
    hyperparameters_knn = dict(
        n_neighbors=list(range(15, 25)), p=[1, 2],
        weights=['uniform', 'distance'],
        metric=['euclidean', 'manhattan', 'minkowski']
    )
    cv_knn = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_knn = GridSearchCV(estimator=knn, param_grid=hyperparameters_knn,
                            n_jobs=-1, cv=cv_knn, scoring='f1', error_score=0)
    best_knn = grid_knn.fit(X_train_scaled, y_train_res)

    # ---- Decision Tree ----
    print("  Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    params_dt = {
        'max_depth': [5, 10, 20, 25],
        'min_samples_leaf': [10, 20, 50, 100, 120],
        'criterion': ["gini", "entropy"]
    }
    grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, cv=4,
                           n_jobs=-1, verbose=0, scoring="accuracy")
    best_dt = grid_dt.fit(X_train_res, y_train_res)

    # ---- Random Forest ----
    print("  Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    params_rf = {
        'n_estimators': [100, 200, 500],
        'max_features': ['sqrt', 'log2']
    }
    cv_rf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, n_jobs=-1,
                           cv=cv_rf, scoring='accuracy', error_score=0)
    best_rf = grid_rf.fit(X_train_res, y_train_res)

    # ---- Logistic Regression ----
    print("  Training Logistic Regression...")
    reg = LogisticRegression(random_state=42)
    reg.fit(X_train_scaled, y_train_res)

    # ---- Naive Bayes ----
    print("  Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_res, y_train_res)

    # Predictions
    predictions = {
        "KNN": (best_knn, best_knn.predict(X_test_scaled), True),
        "Decision Tree": (best_dt, best_dt.predict(X_test), False),
        "Random Forest": (best_rf, best_rf.predict(X_test), False),
        "Logistic Regression": (reg, reg.predict(X_test_scaled), True),
        "Naive Bayes": (nb_model, nb_model.predict(X_test), False),
    }

    # Compute metrics
    model_metrics = []
    for name, (model, pred, needs_scaling) in predictions.items():
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()
        report = classification_report(y_test, pred, output_dict=True)
        model_metrics.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1_score": round(f1_score(y_test, pred), 4),
            "specificity": round(tn / (tn + fp), 4),
            "confusion_matrix": cm.tolist(),
            "best_params": model.best_params_ if hasattr(model, 'best_params_') else {},
            "classification_report": report,
        })

    # SHAP explainer for Random Forest
    print("  Computing SHAP explainer...")
    explainer = shap.TreeExplainer(best_rf.best_estimator_)

    # Store everything
    models_store["knn"] = best_knn
    models_store["decision_tree"] = best_dt
    models_store["random_forest"] = best_rf
    models_store["logistic_regression"] = reg
    models_store["naive_bayes"] = nb_model
    models_store["scaler"] = scaler
    models_store["feature_quantile"] = feature_quantile
    models_store["shap_explainer"] = explainer
    models_store["feature_names"] = X.columns.tolist()
    models_store["imputation_defaults"] = {
        "Glucose": round(float(df['Glucose'].mean()), 2),
        "BloodPressure": round(float(df['BloodPressure'].mean()), 2),
        "SkinThickness": round(float(df['SkinThickness'].median()), 2),
        "Insulin": round(float(df['Insulin'].median()), 2),
        "BMI": round(float(df['BMI'].median()), 2),
    }

    # Extract sample patients directly from the dataset
    sample_patients = []
    for idx, (_, row) in enumerate(df.head(20).iterrows()):
        sample_patients.append({
            "id": idx + 1,
            "pregnancies": float(row["Pregnancies"]),
            "glucose": float(row["Glucose"]),
            "skin_thickness": float(row["SkinThickness"]),
            "bmi": float(row["BMI"]),
            "age": float(row["Age"]),
            "actual_outcome": int(row["Outcome"]),
            "actual_outcome_label": "Diabetic (High Risk)" if row["Outcome"] == 1 else "Non-Diabetic (Low Risk)",
        })

    data_store["sample_patients"] = sample_patients
    data_store["model_metrics"] = model_metrics
    data_store["dataset_info"] = {
        "original_shape": list(original_shape),
        "shape_after_dedup": list(df.shape),
        "zero_counts": zero_counts,
        "summary_statistics": summary_stats,
        "features_used": X.columns.tolist(),
        "target": target_name,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "smote_train_size": int(X_train_res.shape[0]),
        "outcome_distribution": {
            "no_diabetes": int((y == 0).sum()),
            "diabetes": int((y == 1).sum()),
        },
        "imputation_defaults": models_store["imputation_defaults"],
    }

    print("✅ All models trained and ready!")


# ============================================================
# Startup Event
# ============================================================
@app.on_event("startup")
async def startup_event():
    train_all_models()


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "models_loaded": len(models_store) > 0}


@app.get("/api/dataset-info")
async def get_dataset_info():
    """Return dataset statistics and preprocessing info."""
    return data_store.get("dataset_info", {})


@app.get("/api/sample-patients")
async def get_sample_patients():
    """Return real patient records directly from the dataset."""
    return data_store.get("sample_patients", [])


@app.get("/api/models")
async def get_model_metrics():
    """Return performance metrics for all trained models."""
    return data_store.get("model_metrics", [])


@app.get("/api/plots")
async def list_plots():
    """List all available plot files."""
    if not PLOTS_DIR.exists():
        return []
    plots = []
    for f in sorted(PLOTS_DIR.iterdir()):
        if f.suffix == ".png":
            name = f.stem
            # Create a nice title from filename
            title = name.split("_", 1)[1].replace("_", " ").title() if "_" in name else name
            plots.append({"filename": f.name, "title": title, "url": f"/api/plots/{f.name}"})
    return plots


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serve a plot image."""
    plot_path = PLOTS_DIR / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found")
    return FileResponse(str(plot_path), media_type="image/png")


@app.post("/api/predict")
async def predict(patient: PatientInput):
    """
    Predict diabetes risk for a patient using all 5 models.
    Supports either:
    1. Raw clinical values (e.g. Glucose in mg/dL, Age in years, etc.) with automatic
       Quantile Transformation fitted on the dataset.
    2. Normalized quantile values in [0.0, 1.0].
    """
    is_raw = patient.is_raw or patient.glucose > 1.0 or patient.age > 1.0 or patient.bmi > 1.0

    if is_raw:
        imp = models_store["imputation_defaults"]
        raw_p = max(0.0, float(patient.pregnancies))
        raw_g = imp["Glucose"] if patient.glucose <= 0 else float(patient.glucose)
        raw_s = imp["SkinThickness"] if patient.skin_thickness <= 0 else float(patient.skin_thickness)
        raw_b = imp["BMI"] if patient.bmi <= 0 else float(patient.bmi)
        raw_a = max(1.0, float(patient.age))

        raw_features = np.array([[raw_p, raw_g, raw_s, raw_b, raw_a]])
        quantile_features = models_store["feature_quantile"].transform(raw_features)
        features = quantile_features

        raw_display = {
            "pregnancies": round(raw_p, 1),
            "glucose": round(raw_g, 1),
            "skin_thickness": round(raw_s, 1),
            "bmi": round(raw_b, 1),
            "age": round(raw_a, 1),
        }
    else:
        features = np.array([[
            patient.pregnancies,
            patient.glucose,
            patient.skin_thickness,
            patient.bmi,
            patient.age
        ]])
        raw_display = None

    feature_names = models_store["feature_names"]
    patient_df = pd.DataFrame(features, columns=feature_names)
    scaler = models_store["scaler"]
    features_scaled = scaler.transform(features)

    results = []
    model_configs = {
        "KNN": ("knn", True),
        "Decision Tree": ("decision_tree", False),
        "Random Forest": ("random_forest", False),
        "Logistic Regression": ("logistic_regression", True),
        "Naive Bayes": ("naive_bayes", False),
    }

    for display_name, (key, needs_scaling) in model_configs.items():
        model = models_store[key]
        X_input = features_scaled if needs_scaling else features

        proba = model.predict_proba(X_input)[0]
        pred = int(model.predict(X_input)[0])

        results.append(PredictionResult(
            model_name=display_name,
            prediction=pred,
            probability_no_diabetes=round(float(proba[0]), 4),
            probability_diabetes=round(float(proba[1]), 4),
            risk_level="HIGH RISK" if proba[1] >= 0.5 else "LOW RISK"
        ))

    # SHAP explanation using Random Forest
    explainer = models_store["shap_explainer"]
    shap_values = explainer(patient_df)

    # Extract SHAP values for the diabetes class (class 1)
    shap_data = []
    sv = shap_values[0, :, 1]
    for i, fname in enumerate(feature_names):
        shap_data.append({
            "feature": fname,
            "value": round(float(patient_df.iloc[0][fname]), 4),
            "shap_value": round(float(sv.values[i]), 4),
        })

    # Sort by absolute SHAP value
    shap_data.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "is_raw": is_raw,
        "raw_inputs": raw_display,
        "quantile_inputs": {
            "pregnancies": round(float(features[0][0]), 4),
            "glucose": round(float(features[0][1]), 4),
            "skin_thickness": round(float(features[0][2]), 4),
            "bmi": round(float(features[0][3]), 4),
            "age": round(float(features[0][4]), 4),
        },
        "predictions": [r.model_dump() for r in results],
        "shap_explanation": shap_data,
        "base_value": round(float(sv.base_values), 4),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
