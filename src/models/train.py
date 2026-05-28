"""LightGBM baseline training pipeline for store-sales forecasting.

Run from the project root::

    conda activate ds-env
    python src/models/train.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loader import load_raw_data
from src.features.engineering import build_features

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).parents[2]
MODELS_DIR:   Path = PROJECT_ROOT / "models"

# ── Configuration ─────────────────────────────────────────────────────────────

# Last N calendar days of the training set held out for validation
VAL_DAYS: int = 15

# Columns treated as categoricals by LightGBM
CATEGORICAL_FEATURES: List[str] = ["family", "store_type", "holiday_locale"]

# Columns that are identifiers / target — excluded from the feature matrix
_NON_FEATURE_COLS: frozenset = frozenset({"id", "date", "sales"})

# Lag columns whose NaN signals a warmup row to be dropped
_LAG_COLS: List[str] = [
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "onpromotion_lag_7",
]

LGBM_PARAMS: Dict = {
    "objective":         "regression_l2",
    "metric":            "rmse",
    "num_leaves":        127,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "n_estimators":      3_000,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

MLFLOW_EXPERIMENT: str = "store-sales-forecasting"
MODEL_FILENAME:    str = "lgbm_v3.joblib"
_HORIZON_N:        int = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSLE between ground-truth and predicted values.

    Because both arrays are already in log1p space — ``build_features``
    applies ``log1p`` to the sales target — RMSLE collapses to plain RMSE:

    .. math::

        \\text{RMSLE} = \\sqrt{\\frac{1}{n} \\sum_i (\\hat{y}_i - y_i)^2}

    where :math:`y_i = \\log(1 + \\text{sales}_i)`.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values in log1p space.
    y_pred : np.ndarray
        Predicted values in log1p space. Clipped to 0 before scoring to
        avoid negative-prediction artefacts.

    Returns
    -------
    float
        RMSLE score (lower is better).
    """
    y_pred = np.maximum(y_pred, 0.0)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def load_and_build_features() -> pd.DataFrame:
    """Load raw data and return the full engineered feature matrix.

    Calls ``load_raw_data`` followed by ``build_features`` on the training
    split, then drops warmup rows — rows whose lag features are NaN because
    they fall within the first 28 days of each (store_nbr, family) series.

    The temporal cutoff is computed before calling ``build_features`` so that
    target-encoding means are derived exclusively from the training window,
    preventing any leakage from the validation period.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the ``sales`` column in log1p space.
        Warmup rows are excluded; index is reset.
    """
    print("Loading raw data …")
    data = load_raw_data()

    # Raw slice covering only the training window — used for target encoding
    cutoff = data["train"]["date"].max() - pd.Timedelta(days=VAL_DAYS - 1)
    raw_train = data["train"].loc[data["train"]["date"] < cutoff]

    print("\nBuilding features …")
    feat = build_features(
        df           = data["train"],
        stores       = data["stores"],
        oil          = data["oil"],
        holidays     = data["holidays"],
        transactions = data["transactions"],
        train_df     = raw_train,
    )

    n_before = len(feat)
    feat = feat.dropna(subset=_LAG_COLS).reset_index(drop=True)
    print(f"Dropped {n_before - len(feat):,} warmup rows → {len(feat):,} remain")

    return feat


def temporal_split(
    feat: pd.DataFrame,
    val_days: int = VAL_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature matrix into train and validation sets by date.

    The validation set consists of the last *val_days* calendar days in the
    dataset. This mirrors the test period structure and avoids any lookahead.

    Parameters
    ----------
    feat : pd.DataFrame
        Full feature matrix (output of ``load_and_build_features``).
    val_days : int, optional
        Number of trailing calendar days used as validation. Default: 15.

    Returns
    -------
    train_df : pd.DataFrame
        All rows strictly before the validation window.
    val_df : pd.DataFrame
        Rows in the last *val_days* calendar days.
    """
    cutoff = feat["date"].max() - pd.Timedelta(days=val_days - 1)
    train_df = feat.loc[feat["date"] < cutoff]
    val_df   = feat.loc[feat["date"] >= cutoff]

    print(
        f"\nTemporal split:"
        f"\n  train : {train_df['date'].min().date()} → {train_df['date'].max().date()}"
        f"  ({len(train_df):,} rows)"
        f"\n  val   : {val_df['date'].min().date()} → {val_df['date'].max().date()}"
        f"  ({len(val_df):,} rows)"
    )
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of model input columns.

    Excludes identifier and target columns defined in ``_NON_FEATURE_COLS``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix produced by ``build_features``.

    Returns
    -------
    List[str]
        Column names to pass as ``X`` to the model.
    """
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical feature columns to pandas ``category`` dtype.

    LightGBM's sklearn API picks up ``category`` dtype automatically, so
    no ``categorical_feature`` argument is needed in ``fit()``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns listed in ``CATEGORICAL_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with those columns re-cast (in-place, same object).
    """
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def train_model(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    feature_cols: List[str],
) -> lgb.LGBMRegressor:
    """Fit a LightGBM regressor with early stopping on the validation set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition of the feature matrix.
    val_df : pd.DataFrame
        Validation partition of the feature matrix.
    feature_cols : List[str]
        Column names to use as model inputs.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model. The ``best_iteration_`` attribute reflects early
        stopping applied against the validation RMSE.
    """
    X_train = _cast_categoricals(train_df[feature_cols].copy())
    y_train = train_df["sales"].values

    X_val = _cast_categoricals(val_df[feature_cols].copy())
    y_val = val_df["sales"].values

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    print("\nTraining LightGBM …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    return model


def evaluate(
    model: lgb.LGBMRegressor,
    val_df: pd.DataFrame,
    feature_cols: List[str],
) -> float:
    """Compute RMSLE on the validation set and print the result.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Fitted model.
    val_df : pd.DataFrame
        Validation partition.
    feature_cols : List[str]
        Same column list used during training.

    Returns
    -------
    float
        RMSLE score on the validation set.
    """
    X_val = _cast_categoricals(val_df[feature_cols].copy())
    y_val = val_df["sales"].values

    preds = model.predict(X_val)
    score = compute_rmsle(y_val, preds)

    print(f"\nValidation RMSLE : {score:.6f}")
    return score


def train_full(best_iteration: int) -> lgb.LGBMRegressor:
    """Train LightGBM on the complete training set with a fixed iteration count.

    Reloads and engineers features from the full training window
    (2013-01-29 to 2017-08-15) with target encoding derived from all
    available training rows — no validation split is held out.  The
    ``best_iteration`` from the early-stopping run is used directly as
    ``n_estimators`` so the model stops at the same optimal depth without
    needing a validation set.

    The resulting model is intended for Kaggle submission: every training
    row (including the 15-day hold-out used for evaluation) contributes
    to the fitted trees.

    Parameters
    ----------
    best_iteration : int
        Tree count determined by early stopping in the validation-split
        run.  Passed directly as ``n_estimators`` to the new estimator,
        overriding the ``LGBM_PARAMS`` default.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model trained on the complete dataset, saved to
        ``models/lgbm_v3_full.joblib``.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data …")
    data = load_raw_data()

    print("\nBuilding features (full training set) …")
    feat = build_features(
        df           = data["train"],
        stores       = data["stores"],
        oil          = data["oil"],
        holidays     = data["holidays"],
        transactions = data["transactions"],
        train_df     = data["train"],
    )

    n_before = len(feat)
    feat = feat.dropna(subset=_LAG_COLS).reset_index(drop=True)
    print(f"Dropped {n_before - len(feat):,} warmup rows → {len(feat):,} remain")
    print(
        f"Full training window : {feat['date'].min().date()} → "
        f"{feat['date'].max().date()}  ({len(feat):,} rows)"
    )

    feature_cols = get_feature_columns(feat)
    params = {**LGBM_PARAMS, "n_estimators": best_iteration}

    X = _cast_categoricals(feat[feature_cols].copy())
    y = feat["sales"].values

    model = lgb.LGBMRegressor(**params)

    print(f"\nTraining LightGBM on full data (n_estimators={best_iteration}) …")
    model.fit(X, y, callbacks=[lgb.log_evaluation(period=100)])

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="lgbm_v3_full_train"):
        mlflow.log_params({
            **params,
            "n_features":     len(feature_cols),
            "best_iteration": best_iteration,
        })
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"\nMLflow run_id : {mlflow.active_run().info.run_id}")

    model_path = MODELS_DIR / "lgbm_v3_full.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved   : {model_path}")

    return model


def train_horizon_models() -> None:
    """Train 16 direct multi-step LightGBM models, one per forecast horizon.

    For each horizon h ∈ {1 … 16}, the ``sales`` column (already in log1p
    space) is shifted −h rows within each (store_nbr, family) group so the
    model learns to predict log1p(sales at t+h) using only features
    available at time t.  No predicted values are ever fed back as lag
    inputs.

    Validation for horizon h uses the last ``VAL_DAYS`` calendar days of
    the usable feature window — rows whose shifted target is non-NaN.  This
    gives each horizon a clean, forward-looking val set that shrinks by one
    day per unit of h (h=1 → Aug 1-14; h=16 → Jul 16-30).

    Models are logged individually to MLflow (run names
    ``lgbm_horizon_{h}``, tag ``run_group=lgbm_horizon_models``) and saved
    to ``models/horizon/lgbm_h{h}.joblib``.

    Returns
    -------
    None
    """
    horizon_dir = MODELS_DIR / "horizon"
    horizon_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data …")
    data = load_raw_data()

    # Target encoding from pre-val data — use same cutoff as the main model
    te_cutoff = data["train"]["date"].max() - pd.Timedelta(days=VAL_DAYS - 1)
    raw_train = data["train"].loc[data["train"]["date"] < te_cutoff]

    print("\nBuilding features (horizon models) …")
    feat = build_features(
        df           = data["train"],
        stores       = data["stores"],
        oil          = data["oil"],
        holidays     = data["holidays"],
        transactions = data["transactions"],
        train_df     = raw_train,
    )
    n_before = len(feat)
    feat = feat.dropna(subset=_LAG_COLS).reset_index(drop=True)
    print(
        f"Dropped {n_before - len(feat):,} warmup rows → {len(feat):,} remain\n"
        f"Feature window : {feat['date'].min().date()} → {feat['date'].max().date()}"
    )

    feature_cols = get_feature_columns(feat)
    rmsle_scores: List[float] = []

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for h in range(1, _HORIZON_N + 1):
        print(f"\n── h={h:2d}/{_HORIZON_N} " + "─" * 38)

        # Shift target -h within each (store, family); result is log1p(t+h)
        target_h = (
            feat.groupby(["store_nbr", "family"], sort=False)["sales"].shift(-h)
        )
        valid = target_h.notna()

        # Temporal split on the usable rows (non-NaN shifted target)
        max_usable = feat.loc[valid, "date"].max()
        val_cutoff = max_usable - pd.Timedelta(days=VAL_DAYS - 1)
        train_mask = valid & (feat["date"] < val_cutoff)
        val_mask   = valid & (feat["date"] >= val_cutoff)

        print(
            f"  train: {feat.loc[train_mask, 'date'].min().date()}"
            f" → {feat.loc[train_mask, 'date'].max().date()}"
            f"  ({train_mask.sum():,} rows)"
            f"\n  val  : {feat.loc[val_mask, 'date'].min().date()}"
            f" → {feat.loc[val_mask, 'date'].max().date()}"
            f"  ({val_mask.sum():,} rows)"
        )

        X_train = _cast_categoricals(feat.loc[train_mask, feature_cols].copy())
        y_train = target_h.loc[train_mask].values
        X_val   = _cast_categoricals(feat.loc[val_mask, feature_cols].copy())
        y_val   = target_h.loc[val_mask].values

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        rmsle = compute_rmsle(y_val, model.predict(X_val))
        rmsle_scores.append(rmsle)
        print(f"  RMSLE: {rmsle:.6f}  best_iter: {model.best_iteration_}")

        with mlflow.start_run(run_name=f"lgbm_horizon_{h}"):
            mlflow.set_tag("run_group", "lgbm_horizon_models")
            mlflow.log_params({
                **LGBM_PARAMS,
                "horizon":        h,
                "n_features":     len(feature_cols),
                "best_iteration": model.best_iteration_,
            })
            mlflow.log_metric("rmsle_val", rmsle)
            mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = horizon_dir / f"lgbm_h{h}.joblib"
        joblib.dump(model, model_path)

    mean_rmsle = float(np.mean(rmsle_scores))
    print(f"\n{'═' * 50}")
    print("Horizon RMSLE per h:")
    for h, r in enumerate(rmsle_scores, 1):
        print(f"  h={h:2d} : {r:.6f}")
    print(f"  Mean  : {mean_rmsle:.6f}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full LightGBM training pipeline.

    Steps
    -----
    1. Load raw data and engineer features.
    2. Drop warmup rows (NaN lag features).
    3. Temporal train / validation split (last 15 days).
    4. Train LightGBM with early stopping.
    5. Evaluate RMSLE on the validation set.
    6. Log parameters, metrics, and model artifact to MLflow.
    7. Persist the model to ``models/lgbm_v3.joblib``.
    8. Retrain on the full dataset using ``best_iteration_`` as the fixed
       tree count; persist to ``models/lgbm_v3_full.joblib``.
    9. Train 16 direct multi-step horizon models (h=1..16), each saved to
       ``models/horizon/lgbm_h{h}.joblib``.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1–2. Feature matrix ────────────────────────────────────────────────
    feat = load_and_build_features()

    # ── 3. Temporal split ──────────────────────────────────────────────────
    train_df, val_df = temporal_split(feat)
    feature_cols = get_feature_columns(feat)

    print(f"\nFeature columns ({len(feature_cols)}):\n  {feature_cols}")

    # ── 4–5. Train and evaluate ────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="lgbm_v3_target_encoding"):

        model = train_model(train_df, val_df, feature_cols)
        rmsle_val = evaluate(model, val_df, feature_cols)

        # ── 6. MLflow logging ──────────────────────────────────────────────
        logged_params = {
            **LGBM_PARAMS,
            "val_days":       VAL_DAYS,
            "n_features":     len(feature_cols),
            "best_iteration": model.best_iteration_,
        }
        mlflow.log_params(logged_params)
        mlflow.log_metric("rmsle_val", rmsle_val)
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run_id : {run_id}")

    # ── 7. Persist model ───────────────────────────────────────────────────
    model_path = MODELS_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)
    print(f"Model saved   : {model_path}")

    # ── 8. Full-data retrain for Kaggle submission ─────────────────────────
    print(f"\n{'═' * 60}")
    print("Retraining on full dataset …")
    train_full(best_iteration=model.best_iteration_)

    # ── 9. Direct multi-step horizon models ────────────────────────────────
    print(f"\n{'═' * 60}")
    print("Training direct multi-step horizon models (h=1..16) …")
    train_horizon_models()


if __name__ == "__main__":
    main()
