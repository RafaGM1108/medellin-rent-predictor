"""FastAPI application exposing the model."""

from fastapi import FastAPI
from pydantic import BaseModel

from medellin_rent import __version__

app = FastAPI(title="medellin-rent-predictor API", version=__version__)


class Health(BaseModel):
    """Health check response."""

    status: str
    version: str


@app.get("/health")
def health() -> Health:
    """Report that the service is up."""
    return Health(status="ok", version=__version__)


# TODO: add a /predict endpoint that uses medellin_rent.inference.
