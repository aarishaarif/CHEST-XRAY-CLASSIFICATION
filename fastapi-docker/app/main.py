"""FastAPI application for the notebook's two-stage chest X-ray pipeline."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gradcam import GradCAMError, generate_gradcam
from .model import ChestXRayModels, ModelLoadError, load_models
from .preprocessing import MAX_UPLOAD_BYTES, ImageTooLargeError, InvalidImageError, preprocess_image
from .schemas import GradCAMResponse, HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)
CLASS_NAMES = ("COVID", "NORMAL", "PNEUMONIA")
GATEKEEPER_THRESHOLD = 0.5
GRADCAM_IMAGE_TTL_SECONDS = 10 * 60
MAX_STORED_GRADCAM_IMAGES = 256


class GradCAMImageStore:
    """Bounded, thread-safe, expiring in-memory storage for generated PNGs."""

    def __init__(self, ttl_seconds: int, max_items: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._images: dict[str, tuple[float, bytes]] = {}
        self._lock = Lock()

    def _purge_expired(self, now: float) -> None:
        for image_id in [key for key, (expires_at, _) in self._images.items() if expires_at <= now]:
            del self._images[image_id]

    def put(self, image_bytes: bytes) -> str:
        now = monotonic()
        with self._lock:
            self._purge_expired(now)
            while len(self._images) >= self._max_items:
                del self._images[min(self._images, key=lambda key: self._images[key][0])]
            image_id = secrets.token_urlsafe(24)
            self._images[image_id] = (now + self._ttl_seconds, image_bytes)
            return image_id

    def get(self, image_id: str) -> bytes | None:
        with self._lock:
            self._purge_expired(monotonic())
            stored = self._images.get(image_id)
            return None if stored is None else stored[1]


gradcam_image_store = GradCAMImageStore(GRADCAM_IMAGE_TTL_SECONDS, MAX_STORED_GRADCAM_IMAGES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both notebook model artifacts once per application process."""
    app.state.models = None
    app.state.model_error = None
    try:
        app.state.models = await run_in_threadpool(load_models)
        logger.info("Chest X-ray gatekeeper and disease models loaded successfully.")
    except ModelLoadError as exc:
        app.state.model_error = str(exc)
        logger.exception("Chest X-ray models could not be loaded.")
    yield
    app.state.models = None


app = FastAPI(
    title="Chest X-Ray Classification API",
    version="2.0.0",
    description="Two-stage DenseNet121 pipeline: X-ray validation followed by disease classification.",
    lifespan=lifespan,
)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the included browser client."""
    return FileResponse("frontend/index.html")


def get_loaded_models() -> ChestXRayModels:
    if app.state.models is None:
        raise HTTPException(status_code=503, detail="Models are unavailable. Check server logs for details.")
    return app.state.models


async def read_and_preprocess_upload(image: UploadFile | None) -> tuple[bytes, np.ndarray]:
    if image is None or not image.filename:
        raise HTTPException(status_code=400, detail="An image file is required.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image upload exceeds the 10 MiB limit.")
        return image_bytes, preprocess_image(image_bytes)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await image.close()


def _valid_scores(scores: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.shape != shape or not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Model returned invalid prediction scores.")
    return values


def build_prediction(gatekeeper_scores: np.ndarray, disease_scores: np.ndarray | None = None) -> PredictionResponse:
    """Format results exactly as the notebook's gatekeeper decision tree."""
    xray_confidence = float(_valid_scores(gatekeeper_scores, (1,))[0])
    if xray_confidence < GATEKEEPER_THRESHOLD:
        return PredictionResponse(is_xray=False, xray_confidence=xray_confidence, message="This is not an X-ray image.")
    if disease_scores is None:
        raise ValueError("Disease scores are required for an accepted X-ray image.")
    scores = _valid_scores(disease_scores, (len(CLASS_NAMES),))
    predicted_index = int(np.argmax(scores))
    return PredictionResponse(
        is_xray=True,
        xray_confidence=xray_confidence,
        predicted_class=CLASS_NAMES[predicted_index],
        confidence=float(scores[predicted_index]),
        probabilities={name: float(score) for name, score in zip(CLASS_NAMES, scores)},
    )


def run_pipeline(models: ChestXRayModels, batch: np.ndarray) -> PredictionResponse:
    gatekeeper_scores = np.asarray(
        models.gatekeeper.predict(batch, verbose=0)[0],
        dtype=float,
    )

    xray_confidence = float(
        _valid_scores(gatekeeper_scores, (1,))[0]
    )

    # Stage 1: Reject if the image is not an X-ray
    if xray_confidence < GATEKEEPER_THRESHOLD:
        return build_prediction(gatekeeper_scores)

    # Stage 2: Classify accepted X-ray
    disease_scores = np.asarray(
        models.disease_classifier.predict(batch, verbose=0)[0],
        dtype=float,
    )

    return build_prediction(gatekeeper_scores, disease_scores)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health(response: Response) -> HealthResponse:
    if app.state.models is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable")
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(image: UploadFile | None = File(default=None, description="Chest X-ray image file")) -> PredictionResponse:
    """Validate an upload as an X-ray, then classify accepted X-rays."""
    _, batch = await read_and_preprocess_upload(image)
    try:
        return await run_in_threadpool(run_pipeline, get_loaded_models(), batch)
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try another image.") from exc


@app.post("/gradcam", response_model=GradCAMResponse, tags=["Grad-CAM"])
async def gradcam(
    image: UploadFile | None = File(
        default=None,
        description="Chest X-ray image file",
    )
) -> GradCAMResponse:
    """Return a disease Grad-CAM only when the gatekeeper accepts the image as an X-ray."""

    image_bytes, batch = await read_and_preprocess_upload(image)
    models = get_loaded_models()

    try:
        # Step 1: Run the X-ray gatekeeper
        gatekeeper_scores = await run_in_threadpool(
            models.gatekeeper.predict,
            batch,
            verbose=0,
        )

        xray_confidence = float(
            _valid_scores(
                np.asarray(gatekeeper_scores[0], dtype=float),
                (1,),
            )[0]
        )

        # Step 2: If it is not an X-ray, stop here
        if xray_confidence < GATEKEEPER_THRESHOLD:
            return GradCAMResponse(
                is_xray=False,
                xray_confidence=xray_confidence,
                message="This is not an X-ray image.",
            )

        # Step 3: Generate Grad-CAM using the disease classifier
        result = await run_in_threadpool(
            generate_gradcam,
            models.disease_classifier,
            batch,
            image_bytes,
        )

        # Step 4: Build final disease prediction
        prediction = build_prediction(
            np.asarray(gatekeeper_scores[0], dtype=float),
            result.scores,
        )

        # Step 5: Store generated Grad-CAM image
        image_id = gradcam_image_store.put(result.image_bytes)

        return GradCAMResponse(
            **prediction.model_dump(),
            gradcam_image_url=f"/gradcam/image/{image_id}",
        )

    except GradCAMError as exc:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM visualization could not be generated.",
        ) from exc

    except Exception as exc:
        logger.exception("Grad-CAM prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Grad-CAM prediction failed. Please try another image.",
        ) from exc

@app.get("/gradcam/image/{image_id}", response_class=Response, tags=["Grad-CAM"])
def gradcam_image(image_id: str) -> Response:
    image_bytes = gradcam_image_store.get(image_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Grad-CAM image was not found or has expired.")
    return Response(content=image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})
