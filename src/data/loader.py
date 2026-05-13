"""Raw data loading utilities for the store-sales forecasting project."""

from pathlib import Path
from typing import Dict

import pandas as pd


# Columns to parse as datetime, keyed by filename stem
_DATE_COLUMNS: Dict[str, list[str]] = {
    "train": ["date"],
    "test": ["date"],
    "oil": ["date"],
    "holidays_events": ["date"],
    "transactions": ["date"],
}

# Canonical keys returned in the output dict
_FILE_KEYS: Dict[str, str] = {
    "train": "train",
    "test": "test",
    "stores": "stores",
    "oil": "oil",
    "holidays_events": "holidays",
    "transactions": "transactions",
    "sample_submission": "sample_submission",
}


def _print_summary(key: str, df: pd.DataFrame) -> None:
    """Print shape and dtype information for a single DataFrame.

    Parameters
    ----------
    key : str
        Descriptive label shown in the header line.
    df : pd.DataFrame
        DataFrame to summarise.
    """
    print(f"\n{'─' * 50}")
    print(f"  {key.upper()}  |  shape: {df.shape}")
    print(f"{'─' * 50}")
    print(df.dtypes.to_string())


def load_raw_data(
    raw_dir: str | Path = Path(__file__).parents[2] / "data" / "raw",
) -> Dict[str, pd.DataFrame]:
    """Load all raw CSV files for the Corporación Favorita dataset.

    Reads every expected CSV from *raw_dir*, parses known date columns,
    applies forward-fill interpolation to the oil price series, and
    prints a brief shape + dtype summary for each file.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory that contains the raw CSV files.
        Defaults to ``<project_root>/data/raw/``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with the following keys:

        ``"train"``
            Training time series (store_nbr, family, date, sales, onpromotion).
        ``"test"``
            Test period to forecast (same schema minus *sales*).
        ``"stores"``
            Store metadata (city, state, type, cluster).
        ``"oil"``
            Daily Brent crude oil price with forward-filled missing values.
        ``"holidays"``
            National and local holiday/event calendar.
        ``"transactions"``
            Daily transaction counts per store.
        ``"sample_submission"``
            Kaggle submission template.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist or a required CSV is missing.

    Examples
    --------
    >>> data = load_raw_data()
    >>> data["train"].shape
    (3000888, 6)
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    datasets: Dict[str, pd.DataFrame] = {}

    for stem, key in _FILE_KEYS.items():
        csv_path = raw_dir / f"{stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected file not found: {csv_path}")

        parse_dates = _DATE_COLUMNS.get(stem, False)
        df = pd.read_csv(csv_path, parse_dates=parse_dates)

        # Oil prices have sporadic missing values — propagate last known value
        if stem == "oil":
            df = df.sort_values("date").reset_index(drop=True)
            df["dcoilwtico"] = df["dcoilwtico"].ffill()

        datasets[key] = df
        _print_summary(key, df)

    print(f"\nLoaded {len(datasets)} datasets from {raw_dir}\n")
    return datasets
