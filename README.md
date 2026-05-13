# Store Sales Forecasting

Predict unit sales for thousands of store/product-family combinations across Corporación Favorita's retail network in Ecuador. Trained on 4+ years of daily sales data; served as a REST API.

Reference competition: [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

## Project overview

| Stage | Description |
|---|---|
| EDA | Trend, seasonality, promotions, holidays, oil-price correlation |
| Feature engineering | Lag (t-7/14/28), rolling stats, calendar, holiday distances, oil, store metadata |
| Modelling | LightGBM baseline optimised on RMSLE; experiment tracking with MLflow |
| Serving | FastAPI REST API with single-row inference and context-window lag computation |

Metric: **RMSLE** (Root Mean Squared Log Error) — penalises under-predictions more than over-predictions and handles the wide sales variance across product families.

---

## Dataset

Download from Kaggle and place all CSV files in `data/raw/` (do not modify them).

| File | Description |
|---|---|
| `train.csv` | Daily sales per store/family (2013-01-01 to 2017-08-15) |
| `test.csv` | 15-day forecast period to submit |
| `stores.csv` | Store metadata: city, state, type (A–E), cluster (1–17) |
| `oil.csv` | Daily Brent crude oil price (Ecuador is oil-dependent) |
| `holidays_events.csv` | National, regional and local holidays |
| `transactions.csv` | Daily transaction counts per store |
| `sample_submission.csv` | Kaggle submission template |

---

## Stack

| Layer | Libraries |
|---|---|
| Data | pandas 2.3, numpy 2.4 |
| Modelling | lightgbm 4.6, xgboost 3.2, scikit-learn 1.8 |
| Experiment tracking | mlflow 3.12 |
| Serialisation | joblib 1.5 |
| API | fastapi 0.136, uvicorn 0.46, pydantic 2.13 |
| Visualisation | matplotlib 3.10, seaborn 0.13 |
| Notebooks | jupyterlab 4.5, ipykernel 7.2 |
| Runtime | Python 3.11, conda |

---

## Environment setup

```bash
# Create and activate the conda environment
conda create -n ds-env python=3.11 -y
conda activate ds-env

# Install dependencies
pip install -r requirements.txt
```

---

## Project structure

```
store-sales-forecasting/
├── data/
│   ├── raw/              # Kaggle CSVs (download separately — not tracked)
│   └── processed/        # Transformed data ready for modelling
├── notebooks/
│   └── 01_eda.ipynb      # Exploratory data analysis
├── src/
│   ├── data/
│   │   └── loader.py     # load_raw_data() — loads and forward-fills all CSVs
│   ├── features/
│   │   └── engineering.py  # build_features() — lags, rolling stats, calendar, holidays
│   ├── models/
│   │   └── train.py      # LightGBM training pipeline + MLflow logging
│   └── api/
│       ├── main.py       # FastAPI application
│       └── schemas.py    # Pydantic request/response models
├── models/               # Serialised model artefacts (.joblib)
├── tests/
├── requirements.txt
└── README.md
```

---

## Training the model

```bash
conda activate ds-env

# Run the full pipeline: feature engineering → train → evaluate → log → save
python src/models/train.py
```

What it does:
1. Loads all raw CSVs via `load_raw_data()`
2. Builds the feature matrix via `build_features()` (lags, rolling stats, calendar, holidays, oil, store metadata)
3. Drops warmup rows (first 28 days of each store/family series)
4. Temporal split — last 15 days of train used as validation
5. Trains a LightGBM regressor on `log1p(sales)` with early stopping
6. Prints validation RMSLE
7. Logs parameters, metrics, and model artefact to MLflow
8. Saves the model to `models/lgbm_baseline.joblib`

**View experiment results:**

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Running the API

The model must be trained (step above) before starting the server.

```bash
conda activate ds-env
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs are available at `http://localhost:8000/docs`.

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ready",
  "model_loaded": true,
  "date": "2017-08-16",
  "error": null
}
```

### Predict sales

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "store_nbr": 1,
           "family": "GROCERY I",
           "date": "2017-08-16",
           "onpromotion": 0
         }'
```

```json
{
  "store_nbr": 1,
  "family": "GROCERY I",
  "date": "2017-08-16",
  "predicted_sales": 1523.4
}
```

The API builds a 30-day context window of real historical sales for the requested store/family pair, computes lag and rolling features from that history, and returns predictions in original units (log1p is reversed via `expm1` before responding).

---

## Conventions

- **Formatting**: black
- **Linting**: flake8
- **Types**: mypy, type hints required on all public functions
- **Docstrings**: NumPy style
- **Reproducibility**: `random_state=42` everywhere
- **Vectorised operations**: pandas/numpy only — no explicit Python loops over rows
- `data/raw/` is immutable — never modify source files directly
