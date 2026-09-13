"""Loading the two archived Keras models produced by the training notebook."""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "model"
GATEKEEPER_ARCHIVE = MODEL_DIRECTORY / "gatekeeper_best.zip"
DISEASE_ARCHIVE = MODEL_DIRECTORY / "disease_best.zip"


class ModelLoadError(RuntimeError):
    """Raised when a serialized model cannot be reconstructed."""


@dataclass(frozen=True)
class ChestXRayModels:
    """The notebook's two-stage inference pipeline."""

    gatekeeper: keras.Model
    disease_classifier: keras.Model


def load_archive_model(archive_path: Path) -> keras.Model:
    """Rebuild a Keras Functional graph from its archive and load H5 weights."""
    if not archive_path.is_file():
        raise ModelLoadError(f"Model archive was not found: {archive_path}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            required = {"config.json", "model.weights.h5"}
            missing = required.difference(archive.namelist())
            if missing:
                raise ModelLoadError(f"{archive_path.name} is missing: {', '.join(sorted(missing))}")
            model = keras.models.model_from_json(archive.read("config.json").decode("utf-8"))
            with tempfile.TemporaryDirectory(prefix="chest-xray-weights-") as directory:
                weights_path = Path(directory) / "model.weights.h5"
                weights_path.write_bytes(archive.read("model.weights.h5"))
                model.load_weights(weights_path)
    except ModelLoadError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ModelLoadError(f"Could not reconstruct {archive_path.name}.") from exc
    return model


def load_models() -> ChestXRayModels:
    """Load the gatekeeper and final fine-tuned disease classifier at startup."""
    return ChestXRayModels(
        gatekeeper=load_archive_model(GATEKEEPER_ARCHIVE),
        disease_classifier=load_archive_model(DISEASE_ARCHIVE),
    )
