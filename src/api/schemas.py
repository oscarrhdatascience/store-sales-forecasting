"""Pydantic request / response schemas for the store-sales prediction API."""

from datetime import date as date_type

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload for POST /predict.

    Attributes
    ----------
    store_nbr : int
        Store identifier (1–54 in the Corporación Favorita dataset).
    family : str
        Product family name, e.g. ``"GROCERY I"``, ``"BEVERAGES"``.
    date : date
        Forecast date in ISO-8601 format (``YYYY-MM-DD``).
    onpromotion : int
        Number of items in this store/family combination that are on
        promotion on the given date. Defaults to ``0``.
    """

    store_nbr:   int  = Field(..., ge=1, description="Store number (1–54)")
    family:      str  = Field(..., description="Product family, e.g. 'GROCERY I'")
    date:        date_type = Field(..., description="Forecast date (YYYY-MM-DD)")
    onpromotion: int  = Field(default=0, ge=0, description="Items on promotion")

    model_config = {"json_schema_extra": {
        "examples": [{
            "store_nbr": 1,
            "family": "GROCERY I",
            "date": "2017-08-16",
            "onpromotion": 0,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output payload returned by POST /predict.

    Attributes
    ----------
    store_nbr : int
        Echoed from the request.
    family : str
        Echoed from the request.
    date : date
        Echoed from the request.
    predicted_sales : float
        Predicted unit sales for the given store / family / date combination,
        rounded to 2 decimal places. Always ≥ 0.
    """

    store_nbr:       int   = Field(..., description="Store number")
    family:          str   = Field(..., description="Product family")
    date:            date_type = Field(..., description="Forecast date")
    predicted_sales: float = Field(..., ge=0.0, description="Predicted unit sales")
