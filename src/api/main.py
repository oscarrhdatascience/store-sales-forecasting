"""FastAPI application for store-sales predictions.

Start the server::

    conda activate ds-env
    uvicorn src.api.main:app --reload --port 8000

Then POST to http://localhost:8000/predict or visit /docs for the Swagger UI.
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Make src/ importable when the module is run directly
_PROJECT_ROOT = Path(__file__).parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.schemas import PredictionRequest, PredictionResponse
from src.data.loader import load_raw_data
from src.features.engineering import build_features

# Matches CATEGORICAL_FEATURES in train.py — must stay in sync if new
# categorical columns are added to the feature pipeline
_CATEGORICAL_FEATURES: List[str] = ["family", "store_type", "holiday_locale"]

# ── Paths ─────────────────────────────────────────────────────────────────────

_MODEL_PATH: Path = _PROJECT_ROOT / "models" / "lgbm_baseline.joblib"

# ── Application state (populated on startup) ──────────────────────────────────

_state: Dict[str, Any] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and reference data once at startup; release on shutdown."""
    if not _MODEL_PATH.exists():
        # Allow the app to start without a model so /health is still reachable.
        _state["model"] = None
        _state["feature_cols"] = []
        _state["raw_data"] = None
        _state["error"] = f"Model file not found: {_MODEL_PATH}"
    else:
        try:
            _state["model"] = joblib.load(_MODEL_PATH)
            # feature_name_ gives the exact column order the model was trained on
            _state["feature_cols"] = list(_state["model"].feature_name_)
            # train is needed for the context window; the rest for feature joins
            raw = load_raw_data()
            _state["raw_data"] = {k: raw[k] for k in ("train", "stores", "oil", "holidays")}
            _state["error"] = None
        except Exception as exc:  # noqa: BLE001
            _state["model"] = None
            _state["feature_cols"] = []
            _state["raw_data"] = None
            _state["error"] = str(exc)

    yield

    _state.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Sales Forecasting API",
    description=(
        "Predicts unit sales for a Corporación Favorita store/family/date "
        "combination using a LightGBM model trained on RMSLE."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Inference helpers ─────────────────────────────────────────────────────────

_CONTEXT_DAYS: int = 30  # must be ≥ longest lag (28) to populate all lag features


def _build_inference_row(
    request: PredictionRequest,
    raw_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Construct a single-row feature DataFrame ready for model.predict().

    Builds a context window of the last ``_CONTEXT_DAYS`` historical rows for
    the requested (store_nbr, family) pair, appends the inference row (without
    ``sales``), then runs ``build_features`` on the combined DataFrame.
    Because ``sales`` is present in the history rows, ``build_features`` enters
    its lag / rolling branch and computes real lag values for the inference row.
    Only the final row — the inference row — is returned.

    Parameters
    ----------
    request : PredictionRequest
        Validated incoming request.
    raw_data : dict
        Mapping with keys ``"train"``, ``"stores"``, ``"oil"``, ``"holidays"``
        loaded at startup.
    feature_cols : List[str]
        Exact column names and order the model was trained on (from
        ``model.feature_name_``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame aligned to ``feature_cols``, with lag and rolling
        features populated from real historical data.
    """
    pred_date = pd.Timestamp(request.date)

    # Last _CONTEXT_DAYS rows strictly before the prediction date to avoid leakage
    history = (
        raw_data["train"]
        .query(
            "store_nbr == @request.store_nbr"
            " and family == @request.family"
            " and date < @pred_date"
        )
        .nlargest(_CONTEXT_DAYS, "date")
        .sort_values("date")
        [["date", "store_nbr", "family", "sales", "onpromotion"]]
    )

    inference_row = pd.DataFrame([{
        "date":        pred_date,
        "store_nbr":   request.store_nbr,
        "family":      request.family,
        "onpromotion": request.onpromotion,
    }])

    # Concat: history carries `sales`, so pandas fills the inference row with
    # NaN for that column. build_features sees "sales" present and computes
    # lag / rolling features; the inference row gets real lag values.
    combined = pd.concat([history, inference_row], ignore_index=True)

    feat = build_features(
        df       = combined,
        stores   = raw_data["stores"],
        oil      = raw_data["oil"],
        holidays = raw_data["holidays"],
    )

    # Take only the last row — the inference row, now with populated lags
    last_row = feat.iloc[[-1]].copy()

    # Guard: ensure every training feature column is present (e.g. if history
    # was shorter than the longest lag window)
    for col in feature_cols:
        if col not in last_row.columns:
            last_row[col] = np.nan

    # Cast categoricals to match training dtype
    for col in _CATEGORICAL_FEATURES:
        if col in last_row.columns:
            last_row[col] = last_row[col].astype("category")

    return last_row[feature_cols]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health() -> Dict[str, Any]:
    """Return the service status and current UTC date.

    Returns
    -------
    dict
        ``status``: ``"ready"`` or ``"degraded"``
        ``model_loaded``: bool
        ``date``: today's date in ISO-8601
        ``error``: error message if model failed to load, else ``null``
    """
    model_loaded = _state.get("model") is not None
    return {
        "status":       "ready" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "date":         datetime.now(timezone.utc).date().isoformat(),
        "error":        _state.get("error"),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict unit sales",
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict unit sales for a single store / product-family / date combination.

    The model predicts ``log1p(sales)`` internally; this endpoint applies
    ``expm1`` before returning so that ``predicted_sales`` is in original
    units.

    Parameters
    ----------
    request : PredictionRequest
        JSON body with ``store_nbr``, ``family``, ``date``, ``onpromotion``.

    Returns
    -------
    PredictionResponse
        Echo of the request fields plus ``predicted_sales`` (rounded to
        2 decimal places, always ≥ 0).

    Raises
    ------
    HTTPException 503
        If the model has not been loaded (missing joblib file or load error).
    HTTPException 422
        If request validation fails (handled automatically by FastAPI/Pydantic).
    """
    if _state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {_state.get('error', 'unknown error')}",
        )

    X = _build_inference_row(request, _state["raw_data"], _state["feature_cols"])

    log1p_pred = float(_state["model"].predict(X)[0])
    sales = round(float(np.expm1(max(log1p_pred, 0.0))), 2)

    return PredictionResponse(
        store_nbr       = request.store_nbr,
        family          = request.family,
        date            = request.date,
        predicted_sales = sales,
    )
