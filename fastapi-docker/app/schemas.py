"""Response schemas exposed by the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChestXRayClass = Literal["COVID", "NORMAL", "PNEUMONIA"]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]


class PredictionResponse(BaseModel):
    """Result of the notebook's gatekeeper followed by disease classification."""

    is_xray: bool
    xray_confidence: float = Field(ge=0, le=1)
    message: str | None = None
    predicted_class: ChestXRayClass | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[ChestXRayClass, float] | None = None


class GradCAMResponse(PredictionResponse):
    gradcam_image_url: str | None = Field(
        default=None,
        description="Temporary URL that serves the Grad-CAM PNG for accepted X-ray images.",
    )
