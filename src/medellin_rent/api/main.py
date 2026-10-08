"""FastAPI service: predict the monthly rent of an apartment in Medellín.

Run with ``make api`` and open ``/docs``. The model is loaded once at startup from
``paths.models`` (``make train`` creates it); without it ``/predict`` answers 503. With
``conf/rent_index.json`` (``make prices``) predictions are brought to current prices.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from medellin_rent import __version__
from medellin_rent.data.prices import load_rent_index
from medellin_rent.inference.predict import PRICE_REFERENCE_DATE, load_model, predict
from medellin_rent.utils.config import get_config
from medellin_rent.utils.log import get_logger

logger = get_logger(__name__)

ComunaName = Literal[
    "POPULAR",
    "SANTA CRUZ",
    "MANRIQUE",
    "ARANJUEZ",
    "CASTILLA",
    "DOCE DE OCTUBRE",
    "ROBLEDO",
    "VILLA HERMOSA",
    "BUENOS AIRES",
    "LA CANDELARIA",
    "LAURELES",
    "LA AMERICA",
    "SAN JAVIER",
    "EL POBLADO",
    "GUAYABAL",
    "BELEN",
    "PALMITAS",
    "SAN CRISTOBAL",
    "ALTAVISTA",
    "SAN ANTONIO DE PRADO",
    "SANTA ELENA",
]
DATA_PERIOD = "2020-07 to 2021-08"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and rent index once; keep serving /health if the model is missing."""
    paths = get_config().paths
    app.state.rent_index = load_rent_index(paths.rent_index)
    try:
        app.state.model = load_model(paths.models)
    except FileNotFoundError:
        logger.warning("No trained model found: /predict will answer 503 (run `make train`)")
        app.state.model = None
    yield


app = FastAPI(
    title="Medellín Rent Predictor API",
    version=__version__,
    description="Predict the monthly rent (COP) of an apartment in Medellín, Colombia.",
    lifespan=lifespan,
)


class Health(BaseModel):
    """Health check response."""

    status: str
    version: str


class PredictRequest(BaseModel):
    """An apartment to price. Only the comuna is required; unknown fields can be left out."""

    comuna: ComunaName = Field(description="Official comuna or corregimiento name")
    estrato: int | None = Field(None, ge=1, le=6, description="Socioeconomic stratum (1-6)")
    area_m2: float | None = Field(None, ge=10, le=1_000, description="Area in m²")
    bedrooms: int | None = Field(None, ge=0, le=10, description="Number of bedrooms")
    bathrooms: int | None = Field(None, ge=0, le=10, description="Number of bathrooms")
    parking: Literal["yes", "no", "unknown"] = Field(
        "unknown", description="Whether the apartment has parking"
    )
    lat: float | None = Field(None, ge=6.1, le=6.4, description="Latitude (WGS84)")
    lon: float | None = Field(None, ge=-75.75, le=-75.45, description="Longitude (WGS84)")
    has_elevator: bool = Field(False, description="Building has an elevator")
    has_pool: bool = Field(False, description="Building has a pool")
    has_gym: bool = Field(False, description="Building has a gym")
    has_balcony: bool = Field(False, description="Apartment has a balcony")
    has_doorman: bool = Field(False, description="Building has a doorman or security")
    is_furnished: bool = Field(False, description="Apartment is furnished")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "comuna": "EL POBLADO",
                    "estrato": 5,
                    "area_m2": 80,
                    "bedrooms": 2,
                    "bathrooms": 2,
                    "parking": "yes",
                }
            ]
        }
    }


class PriceAdjustment(BaseModel):
    """How the prediction was brought from data-period to current prices."""

    index: str = Field(description="Price index used")
    base_month: str = Field(description="Month of the model's prices (YYYY-MM)")
    current_month: str = Field(description="Month the prediction is brought to (YYYY-MM)")
    factor: float = Field(description="current_index / base_index")
    citation: str = Field(description="Source of the index")


class PredictResponse(BaseModel):
    """Predicted monthly rent with an 80% interval."""

    predicted_rent_cop: float = Field(description="Predicted monthly rent in COP")
    interval_low_cop: float | None = Field(description="Lower end of the interval, COP")
    interval_high_cop: float | None = Field(description="Upper end of the interval, COP")
    interval_level: float | None = Field(description="Coverage of the interval (e.g. 0.8)")
    prices_as_of: str = Field(description="Month (YYYY-MM) of the prices above")
    adjustment: PriceAdjustment | None = Field(
        description="Price adjustment applied, or null if prices are from the data period"
    )
    data_period_rent_cop: float = Field(
        description=f"Prediction at the prices of the training data ({DATA_PERIOD})"
    )
    model: str = Field(description="Model that made the prediction")


def get_model(request: Request) -> tuple[Any, dict[str, Any]]:
    """The loaded model and its metadata, or 503 if there is none."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available: run `make train`")
    return model  # type: ignore[no-any-return]


def get_rent_index(request: Request) -> dict[str, Any] | None:
    """The rent index record loaded at startup, if any."""
    return getattr(request.app.state, "rent_index", None)


@app.get("/health")
def health() -> Health:
    """Report that the service is up."""
    return Health(status="ok", version=__version__)


@app.post("/predict")
def predict_rent(
    listing: PredictRequest,
    loaded: Annotated[tuple[Any, dict[str, Any]], Depends(get_model)],
    rent_index: Annotated[dict[str, Any] | None, Depends(get_rent_index)],
) -> PredictResponse:
    """Predict the monthly rent of one apartment, at current prices when possible."""
    model, metadata = loaded
    row = predict(model, metadata, pd.DataFrame([listing.model_dump()]), rent_index).iloc[0]
    interval = metadata.get("interval") or {}
    suffix = "_current" if rent_index else ""

    def amount(column: str) -> float | None:
        return float(row[column + suffix]) if column + suffix in row else None

    return PredictResponse(
        predicted_rent_cop=float(row["predicted_rent_cop" + suffix]),
        interval_low_cop=amount("interval_low_cop"),
        interval_high_cop=amount("interval_high_cop"),
        interval_level=interval.get("level"),
        prices_as_of=(
            rent_index["current_month"] if rent_index else f"{PRICE_REFERENCE_DATE:%Y-%m}"
        ),
        adjustment=PriceAdjustment(**rent_index) if rent_index else None,
        data_period_rent_cop=float(row["predicted_rent_cop"]),
        model=str(metadata.get("model")),
    )
