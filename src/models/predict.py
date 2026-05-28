"""Generate Kaggle submission (v5) with full-data model and smoothed lag feedback.

Two improvements over v4 (Kaggle score 0.409):

Fix 1 — Full-data model.
    ``lgbm_v3_full.joblib`` is trained on the entire 2013-01-29 to 2017-08-15
    window using ``best_iteration_`` from the early-stopping run as the fixed
    tree count.  Every training row (including the Aug 1-15 validation window
    previously held out) now contributes to the fitted trees.

Fix 2 — Smoothed predicted-lag feedback.
    When a predicted day is appended to the rolling context, its stored sales
    value is blended 50/50 between the raw predicted log1p and the per-
    (store, family) 7-day trailing mean log1p computed once from the original
    real training context.  This reduces the variance of ``sales_lag_7`` for
    test days 8-16, narrowing the covariance gap between the point lag and the
    smoothed rolling features that the model saw during training.

    The final submitted predictions are still the unblended model outputs;
    blending only affects what is stored as context for subsequent lag inputs.

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

_MODEL_FILE:      str = "lgbm_v4_tweedie_full.joblib"
_SUBMISSION_FILE: str = "submission_v8.csv"
_HORIZON_N:       int = 16
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


def _build_store_family_roll_mean(
    context: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    """Compute per-(store_nbr, family) trailing mean log1p sales from context.

    Called once before the iterative prediction loop using the initial real
    training context (no predicted rows).  The resulting baseline is used to
    blend predicted sales stored back into the context, reducing the variance
    of ``sales_lag_7`` for test days 8-16 relative to the smoothed rolling
    features.

    Parameters
    ----------
    context : pd.DataFrame
        Context window DataFrame (real training rows) with columns
        ``store_nbr``, ``family``, ``date``, and ``sales`` (raw scale).
    window : int, optional
        Number of trailing days to average per group. Default: 7.

    Returns
    -------
    pd.DataFrame
        One row per (store_nbr, family) pair with columns ``store_nbr``,
        ``family``, and ``sfm_log1p`` — the trailing-window mean of log1p
        sales used as the smoothing target in the 50/50 blend.
    """
    ctx = context[["store_nbr", "family", "date", "sales"]].copy()
    ctx["sales_log1p"] = np.log1p(ctx["sales"].clip(lower=0))
    ctx = ctx.sort_values(["store_nbr", "family", "date"])
    sfm = (
        ctx.groupby(["store_nbr", "family"])["sales_log1p"]
        .apply(lambda s: s.tail(window).mean())
        .reset_index()
        .rename(columns={"sales_log1p": "sfm_log1p"})
    )
    return sfm


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
    """Predict one test date at a time, feeding smoothed predictions back as lags.

    Each of the 16 calendar days in the test period is processed sequentially:

    1. Append the day's test rows (``sales=NaN`` placeholder) to the running
       context (30 training days + all previously predicted days).
    2. Call ``build_features()`` — lag and rolling features look into the
       context and find log1p of the stored sales, eliminating the NaN cascade
       for test days 8-16.
    3. Recompute Fourier columns against the training reference date
       (2013-01-01) to correct the phase shift from the context window anchor.
    4. Predict in log1p space, clip to >= 0 after expm1.
    5. Store a *blended* sales value in the context — 50% raw prediction and
       50% per-(store, family) 7-day trailing mean from the original real
       training context.  Submitted predictions are always the unblended
       model outputs; blending only smooths the lag inputs for subsequent days.

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

    # Compute per-(store, family) 7-day trailing mean log1p from real context
    # once before the loop — used as the smooth target in the 50/50 blend.
    sfm = _build_store_family_roll_mean(context)

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
            apply_log1p=False,
        )

        # Extract and correct features for the current test date
        test_feat_d = feat[feat["date"] == d].copy()
        test_feat_d = _recompute_fourier_features(test_feat_d, ref_date)

        X_d       = _cast_categoricals(test_feat_d[feature_cols].copy())
        preds_raw = model.predict(X_d).clip(min=0.0)  # Tweedie: already raw scale

        # Convert to log1p for blending and logging only — not for submission.
        preds_log1p = np.log1p(preds_raw)

        # Blend predicted log1p with store-family real-data rolling mean.
        # NaN fallback: use raw prediction for any (store, family) not in sfm.
        sfm_vals = (
            test_feat_d[["store_nbr", "family"]]
            .merge(sfm, on=["store_nbr", "family"], how="left")["sfm_log1p"]
            .to_numpy()
        )
        sfm_vals = np.where(np.isnan(sfm_vals), preds_log1p, sfm_vals)
        blended_log1p = 0.5 * preds_log1p + 0.5 * sfm_vals

        print(
            f"log1p mean = {preds_log1p.mean():.3f} "
            f"(blend={blended_log1p.mean():.3f})  |  "
            f"sales mean = {preds_raw.mean():.1f}"
        )

        # Collect submission rows — unblended predictions, ids from sorted order
        records.append(
            pd.DataFrame({"id": test_feat_d["id"].values, "sales": preds_raw})
        )

        # Store blended raw-scale value in context so lag_7 for subsequent
        # days is smoothed.  build_features() applies log1p internally, so
        # storing expm1(blended_log1p) yields blended_log1p as the lag input.
        predicted_context_rows = test_feat_d[
            ["id", "date", "store_nbr", "family", "onpromotion"]
        ].copy()
        predicted_context_rows["sales"] = np.expm1(blended_log1p)
        context = pd.concat([context, predicted_context_rows], ignore_index=True)

    return (
        pd.concat(records, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )


def predict_horizon(
    test: pd.DataFrame,
    train: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions_aug: pd.DataFrame,
    feature_cols: List[str],
    horizon_dir: Path = None,
) -> pd.DataFrame:
    """Predict all 16 test dates using direct multi-step horizon models.

    Each horizon model h was trained to predict log1p(sales at t+h) using
    only features available at time t.  At inference, t is fixed to the
    last real observation date (2017-08-15): lag and rolling features come
    entirely from real training data, with no predicted values fed back.

    Steps
    -----
    1. Build features on the 30-day real context window (Jul 17 – Aug 15).
    2. Correct Fourier phase against the 2013-01-01 training origin.
    3. Extract the 1 782 feature rows at 2017-08-15 — one per
       (store_nbr, family) — carrying correct lag_7/14/28 and rolling
       stats from real data.
    4. For each horizon h (1 … 16):
       - Load ``{horizon_dir}/lgbm_h{h}.joblib``.
       - Apply model to the Aug-15 snapshot.
       - Look up test-row ids for the target date (Aug 15 + h days).

    Parameters
    ----------
    test : pd.DataFrame
        Raw test DataFrame (id, date, store_nbr, family, onpromotion).
    train : pd.DataFrame
        Full raw training DataFrame — context window source and target-
        encoding reference.
    stores : pd.DataFrame
        Store metadata table.
    oil : pd.DataFrame
        Oil price table with forward-filled values.
    holidays : pd.DataFrame
        Holidays and events table.
    transactions_aug : pd.DataFrame
        Augmented transactions table (real rows up to 2017-08-15 plus
        synthetic constant-mean rows for all test dates).
    feature_cols : List[str]
        Ordered feature column list shared by all 16 horizon models.
    horizon_dir : Path, optional
        Directory containing ``lgbm_h{h}.joblib`` model files.
        Defaults to ``models/horizon``.

    Returns
    -------
    pd.DataFrame
        Submission DataFrame with columns ``id`` and ``sales`` (original
        scale, clipped >= 0), sorted by ``id``.
    """
    if horizon_dir is None:
        horizon_dir = MODELS_DIR / "horizon"
    ref_date    = train["date"].min()
    last_date   = train["date"].max()       # 2017-08-15
    test_dates  = sorted(test["date"].unique())
    n           = len(test_dates)

    # Build features on the 30-day real context (lags are all from real data)
    context = _build_context_window(train)
    feat = build_features(
        df           = context,
        stores       = stores,
        oil          = oil,
        holidays     = holidays,
        transactions = transactions_aug,
        train_df     = train,
    )
    feat = _recompute_fourier_features(feat, ref_date)

    # One row per (store_nbr, family) at the last real date
    base_feat = feat[feat["date"] == last_date].copy()
    print(f"  Base snapshot : {last_date.date()}  ({len(base_feat)} rows)")

    records: List[pd.DataFrame] = []

    for h, d in enumerate(test_dates, 1):
        model = joblib.load(horizon_dir / f"lgbm_h{h}.joblib")

        X           = _cast_categoricals(base_feat[feature_cols].copy())
        preds_log1p = model.predict(X)
        preds_raw   = np.expm1(preds_log1p).clip(min=0.0)

        print(
            f"  h={h:2d}/{n} ({d.date()}) : "
            f"log1p mean = {preds_log1p.mean():.3f}  |  "
            f"sales mean = {preds_raw.mean():.1f}"
        )

        # Match predictions to test ids for this target date
        test_d      = test[test["date"] == d][["id", "store_nbr", "family"]].copy()
        pred_lookup = base_feat[["store_nbr", "family"]].copy()
        pred_lookup["sales"] = preds_raw
        merged = test_d.merge(pred_lookup, on=["store_nbr", "family"], how="left")
        records.append(merged[["id", "sales"]])

    return (
        pd.concat(records, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the v8 Tweedie prediction pipeline and write submission_v8.csv.

    Steps
    -----
    1. Load all raw datasets via ``load_raw_data()``.
    2. Augment the transactions table with per-store synthetic rows for all
       test dates.
    3. Load ``models/lgbm_v4_tweedie_full.joblib`` and its feature column list.
    4. Predict iteratively across the 16 test dates with smoothed lag feedback.
       Features are built in raw (original) scale — no log1p transform —
       matching the Tweedie training configuration.  Model output is already
       in original units; no ``expm1`` reversal is applied.
    5. Save ``id, sales`` CSV to ``data/processed/submission_v8.csv``.
    6. Print a prediction summary: rows, nulls, negatives, min/max/mean,
       and output path.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    data = load_raw_data()

    print("\nAugmenting transactions for test period …")
    test_dates_unique = data["test"]["date"].unique()
    transactions_aug  = _augment_transactions(data["transactions"], test_dates_unique)
    print(
        f"  Original rows : {len(data['transactions']):,}\n"
        f"  Augmented rows: {len(transactions_aug):,}"
    )

    model = joblib.load(MODELS_DIR / _MODEL_FILE)
    feature_cols: List[str] = model.feature_name_

    print("\nIterative daily prediction (16 days, Tweedie) …")
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
