"""Generate Kaggle submission (v4) using iterative daily prediction.

Two bugs from the v3 submission (score 2.70699) are fixed:

Bug 1 — 100% NaN transactions at test time.
    ``transactions.csv`` ends 2017-08-15, so every test row had NaN for the
    three transaction features.  Fix: for each store, compute the last-7-day
    mean and use it as a constant forward-fill proxy across all 16 test dates.

Bug 2 — NaN lag/rolling features for test days 8-16.
    With the 30-day context window, test day 8 (2017-08-23) needs lag_7 from
    2017-08-16 — a test date with no actual sales.  Fix: predict one calendar
    day at a time and feed each day's predicted sales (raw scale) back into
    the context so subsequent days compute correct lag and rolling features.

Run from the project root::

    conda activate ds-env
    python src/models/predict.py
"""

from pathlib import Path
from typing import Any, List
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_raw_data
from src.features.engineering import build_features

# ── Paths and constants ───────────────────────────────────────────────────────

MODELS_DIR:    Path = PROJECT_ROOT / "models"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

_MODEL_FILE:      str = "lgbm_v3.joblib"
_SUBMISSION_FILE: str = "submission_v4.csv"
_CONTEXT_DAYS:    int = 30
_TX_LOOKBACK:     int = 7    # days used for per-store transaction proxy

_CATEGORICAL_FEATURES: List[str] = ["family", "store_type", "holiday_locale"]


# ── Private helpers ───────────────────────────────────────────────────────────

def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical feature columns to pandas ``category`` dtype.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing the categorical columns.

    Returns
    -------
    pd.DataFrame
        Same object with the categorical columns re-cast in-place.
    """
    for col in _CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _build_context_window(
    train: pd.DataFrame,
    context_days: int = _CONTEXT_DAYS,
) -> pd.DataFrame:
    """Extract the last *context_days* calendar days of *train*.

    A 30-day window is sufficient for lag_28 and roll_mean_28 across the
    first 2 test days; longer lags for later test days are populated by the
    iterative prediction loop.

    Parameters
    ----------
    train : pd.DataFrame
        Full raw training DataFrame with a ``date`` column.
    context_days : int, optional
        Number of trailing calendar days to retain. Default: 30.

    Returns
    -------
    pd.DataFrame
        Rows of *train* in ``[max_date - context_days + 1, max_date]``.
    """
    cutoff = train["date"].max() - pd.Timedelta(days=context_days - 1)
    return train.loc[train["date"] >= cutoff].copy()


def _recompute_fourier_features(
    feat: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Correct Fourier columns to use the original training time origin.

    ``build_features()`` anchors Fourier terms to ``df["date"].min()``.  When
    only a context window is passed, that minimum is ~2017-07-17 rather than
    the training origin (2013-01-01), causing a phase discontinuity of roughly
    1.2 rad for the first annual harmonic.  This function recomputes every
    Fourier column using *reference_date* so the values exactly match those
    seen by the model during training.

    Parameters
    ----------
    feat : pd.DataFrame
        Feature matrix with Fourier columns already present; columns are
        overwritten in-place (operates on the caller's copy).
    reference_date : pd.Timestamp
        Time origin used during training — ``train["date"].min()`` (2013-01-01).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with corrected weekly (3 harmonics) and annual
        (6 harmonics) Fourier columns.
    """
    t = (feat["date"] - reference_date).dt.days.to_numpy(dtype=np.float64)

    for k in range(1, 4):
        angle = 2.0 * np.pi * k * t / 7.0
        feat[f"weekly_sin_{k}"] = np.sin(angle)
        feat[f"weekly_cos_{k}"] = np.cos(angle)

    for k in range(1, 7):
        angle = 2.0 * np.pi * k * t / 365.25
        feat[f"annual_sin_{k}"] = np.sin(angle)
        feat[f"annual_cos_{k}"] = np.cos(angle)

    return feat


def _augment_transactions(
    transactions: pd.DataFrame,
    test_dates: np.ndarray,
    lookback_days: int = _TX_LOOKBACK,
) -> pd.DataFrame:
    """Extend the transactions table with synthetic rows for the test period.

    ``transactions.csv`` ends 2017-08-15, so all 16 test dates receive NaN
    for the three transaction features — a distribution never seen during
    training (training rows with NaN transactions were only ~16% of data, not
    100%).  For each store the last *lookback_days* mean is used as a constant
    proxy across all test dates, replacing 100% NaN with a stable,
    store-specific estimate.  The augmented table is passed to
    ``build_features()`` so that ``_add_transactions_features`` computes
    coherent lags and rolling means rather than NaN.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transactions table with columns store_nbr, date, transactions.
    test_dates : np.ndarray
        Array of unique test-period dates (datetime64) to fill.
    lookback_days : int, optional
        Window size (in days) for computing the per-store mean. Default: 7.

    Returns
    -------
    pd.DataFrame
        Augmented transactions table sorted by (store_nbr, date) with one
        synthetic row per (store, test_date) pair appended after the original
        rows.
    """
    tx = transactions[["store_nbr", "date", "transactions"]].copy()
    tx = tx.sort_values(["store_nbr", "date"])

    last_date = tx["date"].max()
    cutoff = last_date - pd.Timedelta(days=lookback_days - 1)
    store_mean = (
        tx.loc[tx["date"] >= cutoff]
        .groupby("store_nbr")["transactions"]
        .mean()
        .reindex(tx["store_nbr"].unique())
        .fillna(tx["transactions"].mean())   # global fallback for sparse stores
    )

    store_arr = store_mean.index.to_numpy()
    txn_arr   = store_mean.values
    n_stores, n_dates = len(store_arr), len(test_dates)

    synthetic = pd.DataFrame({
        "store_nbr":    np.repeat(store_arr, n_dates),
        "date":         np.tile(test_dates, n_stores),
        "transactions": np.repeat(txn_arr,  n_dates),
    })

    return (
        pd.concat([tx, synthetic], ignore_index=True)
        .sort_values(["store_nbr", "date"])
        .reset_index(drop=True)
    )


def predict_iterative(
    test: pd.DataFrame,
    train: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions_aug: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Predict one test date at a time, feeding predictions back as lag inputs.

    Each of the 16 calendar days in the test period is processed sequentially:

    1. Append the day's test rows (``sales=NaN`` placeholder) to the running
       context (30 training days + all previously predicted days with raw-scale
       predicted sales).
    2. Call ``build_features()`` — ``_add_lag_features`` and
       ``_add_rolling_features`` look into the context and find log1p of the
       predicted sales for already-processed days, eliminating the NaN lag
       cascade that caused 56-62% NaN on test days 8-16.
    3. Recompute Fourier columns against the training reference date
       (2013-01-01) to correct the phase shift introduced by the context
       window's ``date.min()`` anchor.
    4. Predict in log1p space, clip to >= 0 after expm1, then append raw-scale
       predictions to the context so the next iteration's lags are correct.

    Note: the ``id``-to-prediction mapping uses ``test_feat_d["id"]``
    (sorted by store_nbr, family as ``build_features`` requires) rather than
    the original ``test_d["id"]`` order, preventing a silent row-mismatch.

    Parameters
    ----------
    test : pd.DataFrame
        Raw test DataFrame (id, date, store_nbr, family, onpromotion).
    train : pd.DataFrame
        Full raw training DataFrame — provides the initial context window and
        the target-encoding reference distribution.
    stores : pd.DataFrame
        Store metadata table (store_nbr, type, city, state, cluster).
    oil : pd.DataFrame
        Oil price table (date, dcoilwtico) with forward-filled values.
    holidays : pd.DataFrame
        Holidays and events table.
    transactions_aug : pd.DataFrame
        Augmented transactions table (real rows up to 2017-08-15, plus
        synthetic constant-mean rows for all test dates).
    model : Any
        Fitted LGBMRegressor exposing a ``predict`` method.
    feature_cols : List[str]
        Ordered feature column list (``model.feature_name_``).

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with columns ``id`` and ``sales`` (original
        scale, clipped to >= 0), sorted by ``id``.
    """
    context  = _build_context_window(train)
    test_dates = sorted(test["date"].unique())
    ref_date   = train["date"].min()
    n = len(test_dates)

    records: List[pd.DataFrame] = []

    for i, d in enumerate(test_dates):
        print(f"  Day {i+1:2d}/{n}: {d.date()} … ", end="", flush=True)

        test_d   = test[test["date"] == d].copy()
        combined = pd.concat(
            [context, test_d.assign(sales=np.nan)], ignore_index=True
        )

        feat = build_features(
            df=combined,
            stores=stores,
            oil=oil,
            holidays=holidays,
            transactions=transactions_aug,
            train_df=train,
        )

        # Extract and correct features for the current test date
        test_feat_d = feat[feat["date"] == d].copy()
        test_feat_d = _recompute_fourier_features(test_feat_d, ref_date)

        X_d        = _cast_categoricals(test_feat_d[feature_cols].copy())
        preds_log1p = model.predict(X_d)
        preds_raw   = np.expm1(preds_log1p).clip(min=0.0)

        print(
            f"log1p mean = {preds_log1p.mean():.3f}  |  "
            f"sales mean = {preds_raw.mean():.1f}"
        )

        # Collect submission rows — ids come from test_feat_d (sorted order)
        records.append(
            pd.DataFrame({"id": test_feat_d["id"].values, "sales": preds_raw})
        )

        # Append raw-scale predictions to context for the next day's lags.
        # build_features() will apply log1p internally, so log1p(expm1(pred))
        # = pred — the lag values for subsequent days are exact.
        predicted_context_rows = test_feat_d[
            ["id", "date", "store_nbr", "family", "onpromotion"]
        ].copy()
        predicted_context_rows["sales"] = np.expm1(preds_log1p)
        context = pd.concat([context, predicted_context_rows], ignore_index=True)

    return (
        pd.concat(records, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the fixed prediction pipeline and write submission_v4.csv.

    Steps
    -----
    1. Load all raw datasets via ``load_raw_data()``.
    2. Load ``lgbm_v3.joblib`` from the models directory.
    3. Augment the transactions table with per-store synthetic rows for all
       test dates (fixes Bug 1 — 100% NaN transactions).
    4. Predict iteratively one calendar day at a time, feeding each day's
       predictions back into the context window (fixes Bug 2 — NaN lag and
       rolling features for test days 8-16).
    5. Save ``id, sales`` CSV to ``data/processed/submission_v4.csv``.
    6. Print a prediction summary: rows, nulls, negatives, min/max/mean,
       and output path.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    data = load_raw_data()

    model_path = MODELS_DIR / _MODEL_FILE
    print(f"\nLoading model : {model_path}")
    model = joblib.load(model_path)
    feature_cols: List[str] = model.feature_name_

    print("\nAugmenting transactions for test period …")
    test_dates_unique = data["test"]["date"].unique()
    transactions_aug  = _augment_transactions(data["transactions"], test_dates_unique)
    print(
        f"  Original rows : {len(data['transactions']):,}\n"
        f"  Augmented rows: {len(transactions_aug):,}"
    )

    print("\nIterative daily prediction (16 days) …")
    submission = predict_iterative(
        test=data["test"],
        train=data["train"],
        stores=data["stores"],
        oil=data["oil"],
        holidays=data["holidays"],
        transactions_aug=transactions_aug,
        model=model,
        feature_cols=feature_cols,
    )

    output_path = PROCESSED_DIR / _SUBMISSION_FILE
    submission.to_csv(output_path, index=False)

    preds = submission["sales"].values
    print(
        f"\nPrediction summary"
        f"\n  Rows       : {len(submission):,}"
        f"\n  Nulls      : {int(submission['sales'].isnull().sum())}"
        f"\n  Negatives  : {int((preds < 0).sum())}"
        f"\n  Sales min  : {preds.min():.4f}"
        f"\n  Sales max  : {preds.max():.4f}"
        f"\n  Sales mean : {preds.mean():.4f}"
        f"\n  Output     : {output_path}"
    )


if __name__ == "__main__":
    main()
