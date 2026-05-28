"""Feature engineering for the store-sales time-series forecasting project."""

from typing import List

import numpy as np
import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

_GROUP_KEYS: List[str] = ["store_nbr", "family"]
_LAG_DAYS: List[int] = [7, 14, 28]
_ROLL_WINDOWS: List[int] = [7, 14, 28]


# ── Private helpers ──────────────────────────────────────────────────────────

def _add_store_features(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Merge store type and cluster from the stores metadata table.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame indexed on (store_nbr, family, date).
    stores : pd.DataFrame
        Stores metadata with columns store_nbr, type, city, state, cluster.

    Returns
    -------
    pd.DataFrame
        Input with new columns ``store_type`` and ``cluster``.
    """
    store_meta = (
        stores[["store_nbr", "type", "cluster"]]
        .rename(columns={"type": "store_type"})
    )
    return df.merge(store_meta, on="store_nbr", how="left")


def _add_oil_features(df: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    """Merge daily oil price and its 7-day lag into the feature matrix.

    The lag is computed on the oil DataFrame directly (date-level signal) so
    the shift is applied in strict calendar order, independent of the
    (store_nbr, family) expansion.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame with a ``date`` column.
    oil : pd.DataFrame
        Oil price table with columns date and dcoilwtico (already forward-filled).

    Returns
    -------
    pd.DataFrame
        Input with new columns ``dcoilwtico`` and ``oil_lag_7``.
    """
    oil_feats = oil[["date", "dcoilwtico"]].copy().sort_values("date")
    oil_feats["oil_lag_7"] = oil_feats["dcoilwtico"].shift(7)
    return df.merge(oil_feats, on="date", how="left")


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar components from the ``date`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        Input with new columns: day_of_week, month, day_of_month,
        week_of_year, is_weekend, quincena.
    """
    dt = df["date"].dt
    df["day_of_week"] = dt.dayofweek.astype(np.int8)       # 0=Mon … 6=Sun
    df["month"] = dt.month.astype(np.int8)
    df["day_of_month"] = dt.day.astype(np.int8)
    df["week_of_year"] = dt.isocalendar().week.astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["quincena"] = np.where(df["day_of_month"] <= 15, 1, 2).astype(np.int8)
    return df


def _add_holiday_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Add holiday indicator, locale, and distance-to-holiday features.

    Only rows where ``transferred == False`` and ``type != 'Work Day'`` are
    treated as actual holidays. Multiple holidays on the same date are
    deduplicated, keeping the first (typically the national-level one).

    Distance features use binary search (``np.searchsorted``) so no Python
    loops are needed regardless of dataset size.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column of dtype datetime64.
    holidays : pd.DataFrame
        Raw holidays_events table from Kaggle.

    Returns
    -------
    pd.DataFrame
        Input with new columns: is_holiday (int8), holiday_locale (str),
        days_to_next_holiday (int), days_since_last_holiday (int).
    """
    real_holidays = (
        holidays
        .query("type != 'Work Day' and transferred == False")
        [["date", "locale"]]
        .drop_duplicates(subset="date", keep="first")
        .rename(columns={"locale": "holiday_locale"})
        .assign(is_holiday=np.int8(1))
    )

    df = df.merge(real_holidays, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(np.int8)
    df["holiday_locale"] = df["holiday_locale"].fillna("None")

    # Vectorised distance features via binary search
    holiday_dates = np.sort(real_holidays["date"].values)
    date_vals = df["date"].values
    n = len(holiday_dates)

    idx_next = np.searchsorted(holiday_dates, date_vals, side="left")

    has_next = idx_next < n
    days_to_next = np.where(
        has_next,
        (holiday_dates[np.minimum(idx_next, n - 1)] - date_vals)
        .astype("timedelta64[D]")
        .astype(np.int16),
        np.int16(999),
    )

    idx_prev = idx_next - 1
    has_prev = idx_prev >= 0
    days_since_last = np.where(
        has_prev,
        (date_vals - holiday_dates[np.maximum(idx_prev, 0)])
        .astype("timedelta64[D]")
        .astype(np.int16),
        np.int16(999),
    )

    df["days_to_next_holiday"] = days_to_next
    df["days_since_last_holiday"] = days_since_last
    return df


def _add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add current and lagged promotion count per (store_nbr, family).

    The 7-day lag is applied within each (store_nbr, family) group so that
    the shift honours series boundaries rather than row position.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns store_nbr, family, and onpromotion. Must be
        sorted by (store_nbr, family, date) before calling.

    Returns
    -------
    pd.DataFrame
        Input with new column ``onpromotion_lag_7``.
    """
    df["onpromotion_lag_7"] = (
        df.groupby(_GROUP_KEYS, sort=False)["onpromotion"].shift(7)
    )
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged sales features per (store_nbr, family).

    Lags are applied after ``sales`` has already been log1p-transformed so
    that lag features and the target share the same scale, which is
    appropriate for RMSLE optimisation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns store_nbr, family, and sales. Must be sorted by
        (store_nbr, family, date) before calling.

    Returns
    -------
    pd.DataFrame
        Input with new columns sales_lag_7, sales_lag_14, sales_lag_28.
    """
    group = df.groupby(_GROUP_KEYS, sort=False)["sales"]
    for lag in _LAG_DAYS:
        df[f"sales_lag_{lag}"] = group.shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and std of sales per (store_nbr, family).

    Windows are computed on sales shifted by 1 (i.e. ending at t-1) to
    prevent any data leakage from the current observation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns store_nbr, family, and sales. Must be sorted by
        (store_nbr, family, date) before calling.

    Returns
    -------
    pd.DataFrame
        Input with new columns sales_roll_mean_{7,14,28} and
        sales_roll_std_{7,14,28}.
    """
    group = df.groupby(_GROUP_KEYS, sort=False)["sales"]
    for window in _ROLL_WINDOWS:
        df[f"sales_roll_mean_{window}"] = group.transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"sales_roll_std_{window}"] = group.transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).std()
        )
    return df


def _add_transactions_features(
    df: pd.DataFrame,
    transactions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge store-level transaction counts and add lag and rolling features.

    Transaction signals are computed at the (store_nbr, date) level before
    merging so that the shift honours strict calendar ordering within each
    store, independent of the (store_nbr, family) expansion in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Main DataFrame with columns store_nbr and date.
    transactions : pd.DataFrame or None
        Daily transaction counts with columns store_nbr, date, transactions.
        When None, returns ``df`` unchanged (inference mode where transaction
        data is unavailable).

    Returns
    -------
    pd.DataFrame
        Input with new columns ``transactions``, ``transactions_lag_7``, and
        ``transactions_roll_mean_7``. When *transactions* is None, returns the
        input unchanged.
    """
    if transactions is None:
        return df

    tx = (
        transactions[["store_nbr", "date", "transactions"]]
        .copy()
        .sort_values(["store_nbr", "date"])
    )
    store_grp = tx.groupby("store_nbr", sort=False)["transactions"]
    tx["transactions_lag_7"] = store_grp.shift(7)
    tx["transactions_roll_mean_7"] = store_grp.transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    return df.merge(
        tx[["store_nbr", "date", "transactions",
            "transactions_lag_7", "transactions_roll_mean_7"]],
        on=["store_nbr", "date"],
        how="left",
    )


def _add_earthquake_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary flag for the 2016 Ecuador earthquake recovery period.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        Input with new column ``is_earthquake_period`` (int8): 1 for dates
        between 2016-04-16 and 2016-05-16 inclusive, 0 otherwise.
    """
    _QUAKE_START = pd.Timestamp("2016-04-16")
    _QUAKE_END = pd.Timestamp("2016-05-16")
    df["is_earthquake_period"] = (
        (df["date"] >= _QUAKE_START) & (df["date"] <= _QUAKE_END)
    ).astype(np.int8)
    return df


def _add_fourier_features(
    df: pd.DataFrame,
    period: float,
    n_terms: int,
    prefix: str,
) -> pd.DataFrame:
    """Add sine and cosine Fourier terms to capture cyclic seasonality.

    The time index is the number of elapsed days from ``date.min()``, so the
    Fourier phase is dataset-relative and does not depend on calendar year.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column of dtype datetime64.
    period : float
        Length of the seasonal cycle in days (e.g. 7.0 for weekly,
        365.25 for annual).
    n_terms : int
        Number of Fourier harmonics to include. Produces 2 * n_terms columns.
    prefix : str
        Column name prefix; e.g. ``"weekly"`` yields ``weekly_sin_1``,
        ``weekly_cos_1``, …, ``weekly_sin_n``, ``weekly_cos_n``.

    Returns
    -------
    pd.DataFrame
        Input with 2 * n_terms new float columns appended.
    """
    t = (df["date"] - df["date"].min()).dt.days.to_numpy(dtype=np.float64)
    for k in range(1, n_terms + 1):
        angle = 2.0 * np.pi * k * t / period
        df[f"{prefix}_sin_{k}"] = np.sin(angle)
        df[f"{prefix}_cos_{k}"] = np.cos(angle)
    return df


def _add_target_encoding(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add mean-sales target encoding per family, store, and (store, family).

    Means are computed exclusively from *train_df* to prevent leakage from the
    validation or test period. The ``sales`` column in *train_df* is
    log1p-transformed internally so the encoded features are on the same scale
    as the target. Unseen combinations at inference time receive the global
    mean computed from *train_df*.

    Parameters
    ----------
    df : pd.DataFrame
        Main feature matrix to augment. Must contain columns family and
        store_nbr.
    train_df : pd.DataFrame or None
        Raw training-period DataFrame with columns family, store_nbr, and
        sales (in original scale). When None, returns ``df`` unchanged.

    Returns
    -------
    pd.DataFrame
        Input with new columns ``family_mean_sales``, ``store_mean_sales``,
        and ``store_family_mean_sales``.
    """
    if train_df is None:
        return df

    enc = train_df[["family", "store_nbr"]].copy()
    enc["sales_log1p"] = np.log1p(train_df["sales"].values)
    global_mean: float = enc["sales_log1p"].mean()

    family_mean = (
        enc.groupby("family")["sales_log1p"].mean().rename("family_mean_sales")
    )
    store_mean = (
        enc.groupby("store_nbr")["sales_log1p"].mean().rename("store_mean_sales")
    )
    store_family_mean = (
        enc.groupby(["store_nbr", "family"])["sales_log1p"]
        .mean()
        .rename("store_family_mean_sales")
    )

    df = df.merge(family_mean, on="family", how="left")
    df = df.merge(store_mean, on="store_nbr", how="left")
    df = df.merge(store_family_mean, on=["store_nbr", "family"], how="left")

    df["family_mean_sales"] = df["family_mean_sales"].fillna(global_mean)
    df["store_mean_sales"] = df["store_mean_sales"].fillna(global_mean)
    df["store_family_mean_sales"] = df["store_family_mean_sales"].fillna(global_mean)

    return df


# ── Public API ───────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
    apply_log1p: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix for training or inference.

    Applies all feature groups in the correct order:

    1.  Sort by (store_nbr, family, date) — mandatory before any lag/roll.
    2.  Store metadata (type, cluster).
    3.  Oil price and its 7-day lag (date-level join).
    4.  Calendar decomposition.
    5.  Holiday indicator, locale, and distance features.
    6.  Promotion current + 7-day lag.
    7.  Transaction count, 7-day lag, 7-day rolling mean (skipped when
        *transactions* is None).
    8.  Earthquake recovery period flag (2016-04-16 – 2016-05-16).
    9.  Weekly Fourier terms (period=7, n_terms=3).
    10. Annual Fourier terms (period=365.25, n_terms=6).
    11. Target encoding — mean log1p(sales) per family, store, and
        (store, family) derived from *train_df* only (skipped when None).
    12. Optional log1p transform on ``sales`` (controlled by *apply_log1p*).
    13. Lag features on ``sales``: t-7, t-14, t-28.
    14. Rolling mean and std on ``sales``: 7-, 14-, 28-day windows.

    When *apply_log1p* is ``True`` (default), log1p is applied before lags and
    rolling stats so all sales-derived features share the same scale as the
    log1p target — appropriate for RMSE-on-log1p optimisation.  Set
    *apply_log1p* to ``False`` when using objectives that operate on the
    original scale (e.g. Tweedie), so that lag and rolling features are also
    in original units.

    Parameters
    ----------
    df : pd.DataFrame
        Raw train or test DataFrame loaded by ``load_raw_data()``. Expected
        columns: id, date, store_nbr, family, sales (train only), onpromotion.
    stores : pd.DataFrame
        Stores metadata table (store_nbr, type, city, state, cluster).
    oil : pd.DataFrame
        Oil price table (date, dcoilwtico) with forward-filled missing values.
    holidays : pd.DataFrame
        Holidays and events table (date, type, locale, transferred, …).
    transactions : pd.DataFrame or None, optional
        Daily transaction counts (store_nbr, date, transactions). When None
        (default), transaction features are skipped — suitable for inference
        where this signal is unavailable.
    train_df : pd.DataFrame or None, optional
        Raw training-period DataFrame (family, store_nbr, sales in original
        scale) used exclusively to compute target-encoding means. Must cover
        only the training window (no validation or test rows) to prevent
        leakage. When None (default), target encoding is skipped.
    apply_log1p : bool, optional
        Whether to apply ``log1p`` to the ``sales`` column before computing
        lag and rolling features. Default ``True`` (log1p space, RMSE
        optimisation). Pass ``False`` for Tweedie or other objectives that
        train directly on original-scale sales.

    Returns
    -------
    pd.DataFrame
        Feature matrix sorted by (store_nbr, family, date) with all engineered
        columns appended. The ``sales`` column — when present — is returned in
        log1p space when *apply_log1p* is ``True``, or in original scale when
        ``False``. NaN values in lag / rolling columns are expected for the
        earliest dates in each series and must be handled downstream
        (e.g. dropped during training or imputed).

    Notes
    -----
    The input DataFrames are never mutated; all operations work on copies.

    Examples
    --------
    >>> from src.data.loader import load_raw_data
    >>> from src.features.engineering import build_features
    >>> data = load_raw_data()
    >>> feat = build_features(
    ...     data["train"], data["stores"], data["oil"], data["holidays"]
    ... )
    >>> feat.shape[1] > data["train"].shape[1]
    True
    """
    out = df.copy()

    # Step 1 — canonical sort order; all group-aware operations rely on this
    out = out.sort_values([*_GROUP_KEYS, "date"]).reset_index(drop=True)

    # Steps 2–6 — joins and deterministic transforms (order-independent)
    out = _add_store_features(out, stores)
    out = _add_oil_features(out, oil)
    out = _add_calendar_features(out)
    out = _add_holiday_features(out, holidays)
    out = _add_promotion_features(out)

    # Steps 7–11 — context signals, Fourier seasonality, target encoding
    out = _add_transactions_features(out, transactions)
    out = _add_earthquake_feature(out)
    out = _add_fourier_features(out, period=7.0, n_terms=3, prefix="weekly")
    out = _add_fourier_features(out, period=365.25, n_terms=6, prefix="annual")
    out = _add_target_encoding(out, train_df)

    # Steps 12–14 — sales-derived features; only available on train-like data
    if "sales" in out.columns:
        if apply_log1p:
            out["sales"] = np.log1p(out["sales"])
        out = _add_lag_features(out)
        out = _add_rolling_features(out)

    return out
