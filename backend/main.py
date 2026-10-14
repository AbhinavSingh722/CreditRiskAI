"""
FastAPI Credit Risk Assessment Backend
=======================================
IBM SkillsBuild / BharatCares / AICTE Internship 2026
Student: Abhinav Singh | ID: IBMUEDA3522
"""

from __future__ import annotations

import io
import logging
import os
import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MODEL_PATH   = Path(os.getenv("MODEL_PATH",   "./models/pipeline_bundle.pkl"))
HOST         = os.getenv("HOST",   "0.0.0.0")
PORT         = int(os.getenv("PORT", "8000"))
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
MODEL_VERSION = "1.0.0"

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_risk_api")

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
_BUNDLE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _BUNDLE
    if not MODEL_PATH.exists():
        logger.error("Model bundle not found at %s", MODEL_PATH)
        raise RuntimeError(f"Model bundle missing: {MODEL_PATH}")
    _BUNDLE = joblib.load(MODEL_PATH)
    logger.info("Model bundle loaded from %s", MODEL_PATH)
    logger.info("Model name: %s | training date: %s",
                _BUNDLE.get("model_name"), _BUNDLE.get("training_date"))
    yield


app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit default risk scoring with SHAP-based explanations.",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# MODEL HELPERS
# ──────────────────────────────────────────────
def _load_bundle_if_needed() -> None:
    """Lazy-load the model bundle (supports TestClient without lifespan trigger)."""
    global _BUNDLE
    if not _BUNDLE:
        _mp = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
        if _mp.exists():
            _BUNDLE = joblib.load(_mp)
            logger.info("Model bundle lazy-loaded from %s", _mp)
        else:
            logger.error("Model bundle not found at %s", _mp)


def _get_bundle() -> Dict[str, Any]:
    _load_bundle_if_needed()
    if not _BUNDLE:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _BUNDLE


# ──────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ──────────────────────────────────────────────
class ApplicantFeatures(BaseModel):
    """Input features for a single applicant prediction."""
    AMT_INCOME_TOTAL:     float = Field(..., gt=0, description="Annual income (USD)")
    AMT_CREDIT:           float = Field(..., gt=0, description="Loan amount (USD)")
    AMT_ANNUITY:          float = Field(..., gt=0, description="Monthly annuity (USD)")
    DAYS_BIRTH:           float = Field(..., lt=0, description="Days since birth (negative)")
    DAYS_EMPLOYED:        float = Field(default=-365, description="Days employed (negative, or 365243 if unemployed)")
    EXT_SOURCE_1:         Optional[float] = Field(default=0.5)
    EXT_SOURCE_2:         Optional[float] = Field(default=0.5)
    EXT_SOURCE_3:         Optional[float] = Field(default=0.5)
    BUREAU_LOAN_COUNT:    Optional[float] = Field(default=0.0)
    BUREAU_AVG_DAYS_PAST_DUE: Optional[float] = Field(default=0.0)
    BUREAU_TOTAL_CREDIT_SUM:  Optional[float] = Field(default=0.0)

    @field_validator("DAYS_BIRTH")
    @classmethod
    def days_birth_must_be_negative(cls, v):
        if v > 0:
            raise ValueError("DAYS_BIRTH must be negative (days before today)")
        return v

    class Config:
        extra = "allow"  # allow additional features


class PredictionResponse(BaseModel):
    sk_id_curr:          Optional[int]
    default_probability: float
    risk_band:           str
    cluster_id:          int
    persona_name:        str
    explanation:         str
    adverse_action_codes: List[str]


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name:     str
    training_date:  str
    version:        str
    n_features:     int
    feature_list:   List[str]
    optimal_threshold: float


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
RISK_BANDS = [
    (0.00, 0.20, "Very Low"),
    (0.20, 0.40, "Low"),
    (0.40, 0.60, "Medium"),
    (0.60, 0.80, "High"),
    (0.80, 1.01, "Very High"),
]


def _risk_band(prob: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= prob < hi:
            return label
    return "Very High"


def _engineer_features(data: dict) -> dict:
    """Reproduce the same feature engineering done in the notebook."""
    eps = 1e-6
    data = dict(data)
    income = data.get("AMT_INCOME_TOTAL", 1)
    credit = data.get("AMT_CREDIT", 1)
    annuity = data.get("AMT_ANNUITY", 1)
    days_emp = data.get("DAYS_EMPLOYED", -365)
    days_birth = data.get("DAYS_BIRTH", -10000)

    # Fix anomaly
    if days_emp == 365243:
        days_emp = np.nan
    data["DAYS_EMPLOYED"] = days_emp if days_emp is not None else 0.0

    data["CREDIT_INCOME_RATIO"]  = credit / (income + eps)
    data["ANNUITY_INCOME_RATIO"] = annuity / (income + eps)
    data["CREDIT_TERM"]          = annuity / (credit + eps)
    data["EMPLOYED_BIRTH_RATIO"] = (days_emp or 0) / (days_birth + eps)
    ext = [data.get(f"EXT_SOURCE_{i}", 0.5) or 0.5 for i in (1, 2, 3)]
    data["EXT_SOURCE_MEAN"] = float(np.mean(ext))
    data["EXT_SOURCE_STD"]  = float(np.std(ext))
    data["DAYS_EMPLOYED_ANOM"] = 1 if days_emp in (365243, None) else 0
    return data


def _build_feature_vector(data: dict, feature_names: List[str]) -> np.ndarray:
    engineered = _engineer_features(data)
    vec = np.array([engineered.get(f, 0.0) for f in feature_names], dtype=np.float64)
    return vec.reshape(1, -1)


def _get_explanation_and_codes(
    feature_vec: np.ndarray,
    feature_names: List[str],
    adverse_codes: dict,
    model,
    n_top: int = 3,
) -> tuple[str, list[str]]:
    """Generate explanation using SHAP if available, else fallback."""
    try:
        import shap
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(feature_vec)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]
        except Exception:
            background = np.zeros((1, len(feature_names)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            sv = explainer.shap_values(feature_vec, nsamples=50)[0][:, 1]

        contrib = dict(zip(feature_names, sv))
        sorted_c = sorted(contrib.items(), key=lambda x: x[1], reverse=True)
        top_pos = sorted_c[:n_top]
        top_neg = sorted_c[-n_top:]

        lines = ["Risk-increasing factors:"]
        for feat, val in top_pos:
            desc = adverse_codes.get(feat, feat.replace("_", " ").title())
            lines.append(f"  (+{val:.3f}) {desc}")
        lines.append("Risk-reducing factors:")
        for feat, val in top_neg:
            desc = adverse_codes.get(feat, feat.replace("_", " ").title())
            lines.append(f"  ({val:.3f}) {desc}")

        explanation = "\n".join(lines)
        top_risk = sorted_c[:n_top]
        adverse = [
            f"Reason {i+1}: {adverse_codes.get(f, f.replace('_',' ').title())}"
            for i, (f, _) in enumerate(top_risk)
        ]
        return explanation, adverse

    except ImportError:
        pass

    # Fallback: feature importance proxy
    try:
        fi = dict(zip(feature_names, model.feature_importances_))
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:n_top]
        explanation = "Top risk factors (feature importance):\n" + "\n".join(
            f"  {adverse_codes.get(f, f)}: importance={v:.4f}" for f, v in sorted_fi
        )
        adverse = [
            f"Reason {i+1}: {adverse_codes.get(f, f.replace('_',' ').title())}"
            for i, (f, _) in enumerate(sorted_fi)
        ]
        return explanation, adverse
    except Exception:
        return "Explanation unavailable.", []


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check():
    """Liveness check."""
    bundle = _get_bundle()
    return HealthResponse(
        status="ok",
        model=bundle.get("model_name", "unknown"),
        version=MODEL_VERSION,
        timestamp=datetime.datetime.utcnow().isoformat(),
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Operations"])
def model_info():
    """Model metadata: version, training date, feature list."""
    bundle = _get_bundle()
    feats = bundle.get("selected_features", [])
    return ModelInfoResponse(
        model_name=bundle.get("model_name", "unknown"),
        training_date=bundle.get("training_date", "unknown"),
        version=MODEL_VERSION,
        n_features=len(feats),
        feature_list=feats,
        optimal_threshold=bundle.get("optimal_threshold", 0.5),
    )


@app.get("/clusters", tags=["Insights"])
def cluster_summary():
    """Borrower persona / cluster summary statistics."""
    bundle = _get_bundle()
    names = bundle.get("cluster_names", {})
    profile = bundle.get("cluster_profile", {})
    return {"cluster_names": names, "cluster_profile": profile}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(applicant: ApplicantFeatures, sk_id_curr: Optional[int] = None):
    """Score a single applicant. Returns probability, risk band, persona, explanation."""
    logger.info("Predict request: sk_id_curr=%s", sk_id_curr)
    bundle = _get_bundle()

    model           = bundle["model"]
    feature_names   = bundle["selected_features"]
    cluster_model   = bundle["cluster_kmeans"]
    cluster_scaler  = bundle["cluster_scaler"]
    cluster_feats   = bundle["cluster_features"]
    cluster_names   = bundle["cluster_names"]
    threshold       = bundle["optimal_threshold"]
    adverse_codes   = bundle.get("adverse_action_codes", {})

    data = applicant.dict()
    data = _engineer_features(data)

    # Predict probability
    X = _build_feature_vector(data, feature_names)
    try:
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Validate probability range
    if not (0.0 <= prob <= 1.0):
        raise HTTPException(status_code=500, detail=f"Invalid probability: {prob}")

    band = _risk_band(prob)

    # Cluster assignment
    try:
        cl_vec = np.array([data.get(f, 0.0) for f in cluster_feats], dtype=np.float64).reshape(1, -1)
        cl_vec_s = cluster_scaler.transform(cl_vec)
        cluster_id = int(cluster_model.predict(cl_vec_s)[0])
        persona = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    except Exception as e:
        logger.warning("Cluster assignment failed: %s", e)
        cluster_id, persona = 0, "Unknown"

    # Explanation & adverse-action codes
    explanation, aac = _get_explanation_and_codes(X, feature_names, adverse_codes, model)
    if prob < threshold:
        aac = []  # only show adverse codes for high-risk

    logger.info("Prediction: prob=%.4f band=%s cluster=%s", prob, band, cluster_id)
    return PredictionResponse(
        sk_id_curr=sk_id_curr,
        default_probability=round(prob, 6),
        risk_band=band,
        cluster_id=cluster_id,
        persona_name=persona,
        explanation=explanation,
        adverse_action_codes=aac,
    )


@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept a CSV upload of multiple applicants.
    Returns scored CSV/JSON with default_probability, risk_band, persona, explanation.
    """
    logger.info("Batch predict request: filename=%s", file.filename)
    bundle = _get_bundle()

    content = await file.read()
    try:
        df_in = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if df_in.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    required = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    model         = bundle["model"]
    feature_names = bundle["selected_features"]
    cluster_model = bundle["cluster_kmeans"]
    cluster_scaler= bundle["cluster_scaler"]
    cluster_feats = bundle["cluster_features"]
    cluster_names = bundle["cluster_names"]
    adverse_codes = bundle.get("adverse_action_codes", {})

    results = []
    for _, row in df_in.iterrows():
        data = row.to_dict()
        data = _engineer_features(data)
        X = _build_feature_vector(data, feature_names)
        prob = float(model.predict_proba(X)[0, 1])
        band = _risk_band(prob)
        try:
            cl_vec = np.array([data.get(f, 0.0) for f in cluster_feats], dtype=np.float64).reshape(1, -1)
            cluster_id = int(cluster_model.predict(cluster_scaler.transform(cl_vec))[0])
            persona = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        except Exception:
            cluster_id, persona = 0, "Unknown"
        explanation, aac = _get_explanation_and_codes(X, feature_names, adverse_codes, model)
        results.append({
            "SK_ID_CURR":          row.get("SK_ID_CURR"),
            "default_probability": round(prob, 6),
            "risk_band":           band,
            "cluster_id":          cluster_id,
            "persona_name":        persona,
            "top_reason":          aac[0] if aac else "",
        })

    df_out = pd.DataFrame(results)
    logger.info("Batch predict complete: %d rows", len(df_out))
    return df_out.to_dict(orient="records")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
