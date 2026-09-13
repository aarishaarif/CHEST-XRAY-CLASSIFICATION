"""API tests using deterministic stand-ins for the two trained models."""

from __future__ import annotations

from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import Image

import app.main as main
from app.gradcam import GradCAMResult
from app.model import ChestXRayModels


class FakeGatekeeper:
    def __init__(self, score: float = 0.91) -> None:
        self.score = score

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[self.score]], dtype=np.float32)


class FakeDiseaseClassifier:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.02, 0.03, 0.95]], dtype=np.float32)


def valid_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def image_upload(content: bytes, content_type: str = "image/png") -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("xray.png", content, content_type)}


@pytest.fixture()
def api_app():
    main.app.state.models = ChestXRayModels(FakeGatekeeper(), FakeDiseaseClassifier())
    yield main.app


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_when_models_are_available(api_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://testserver") as client:
        response = await client.get("/health")
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_predict_rejects_non_xray(api_app) -> None:
    main.app.state.models = ChestXRayModels(FakeGatekeeper(0.2), FakeDiseaseClassifier())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://testserver") as client:
        response = await client.post("/predict", files=image_upload(valid_image_bytes()))
    assert response.status_code == 200
    assert response.json()["is_xray"] is False
    assert response.json()["message"] == "This is not an X-ray image."
    assert response.json()["predicted_class"] is None


@pytest.mark.anyio
async def test_predict_returns_two_stage_result(api_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://testserver") as client:
        response = await client.post("/predict", files=image_upload(valid_image_bytes()))
    assert response.status_code == 200
    assert response.json()["is_xray"] is True
    assert response.json()["predicted_class"] == "PNEUMONIA"
    assert response.json()["probabilities"]["PNEUMONIA"] == pytest.approx(0.95)


@pytest.mark.anyio
async def test_gradcam_returns_image_for_xray(api_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "generate_gradcam", lambda *args: GradCAMResult(np.array([0.02, 0.03, 0.95]), b"fake-png"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://testserver") as client:
        response = await client.post("/gradcam", files=image_upload(valid_image_bytes()))
        image_response = await client.get(response.json()["gradcam_image_url"])
    assert response.status_code == 200
    assert response.json()["is_xray"] is True
    assert image_response.content == b"fake-png"


@pytest.mark.anyio
async def test_predict_rejects_invalid_image(api_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api_app), base_url="http://testserver") as client:
        response = await client.post("/predict", files=image_upload(b"not an image"))
    assert response.status_code == 422
