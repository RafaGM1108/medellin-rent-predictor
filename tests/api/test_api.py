from fastapi.testclient import TestClient

from medellin_rent import __version__
from medellin_rent.api.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
