# Chest X-Ray Classification API

An end-to-end chest X-ray classification project built with TensorFlow, FastAPI, and a browser-based interface. The application first checks whether an uploaded image is a chest X-ray and then classifies accepted images as `COVID`, `NORMAL`, or `PNEUMONIA`.

## Features

- Two-stage prediction pipeline to reject non-X-ray uploads before disease classification.
- DenseNet121-based gatekeeper and disease-classification models.
- Three disease classes: `COVID`, `NORMAL`, and `PNEUMONIA`.
- FastAPI endpoints for prediction, health checks, and Grad-CAM visualizations.
- Browser UI with image preview, prediction probabilities, and an explainability view.
- Input validation for image type, file size, and decoded image dimensions.
- Docker and Docker Compose support.

## Model Architecture

The notebook trains a two-stage DenseNet121 pipeline at an input size of **224 × 224 RGB**.

### Stage 1: X-Ray Gatekeeper

The gatekeeper separates chest X-rays from natural images before the disease model is called.

```text
Input image
  -> resize to 224 × 224 RGB
  -> DenseNet121 (ImageNet weights, frozen during training)
  -> GlobalAveragePooling2D
  -> Dense(128, ReLU)
  -> Dropout(0.3)
  -> Dense(1, Sigmoid)
  -> X-ray confidence
```

- Positive examples: chest X-rays from the training dataset.
- Negative examples: sampled CIFAR-10 natural images.
- Decision threshold: `0.5`.
- If the score is below `0.5`, the API returns `This is not an X-ray image.`

### Stage 2: Disease Classifier

Only images accepted by the gatekeeper are passed to the disease classifier.

```text
Accepted chest X-ray
  -> DenseNet121 (ImageNet weights)
  -> GlobalAveragePooling2D
  -> Dense(256, ReLU)
  -> Dropout(0.4)
  -> Dense(3, Softmax)
  -> COVID / NORMAL / PNEUMONIA
```

Training runs in two phases:

1. Train the new classification head with the DenseNet121 backbone frozen.
2. Fine-tune the final 30 backbone layers and keep the checkpoint with the best validation accuracy.

The notebook uses DenseNet preprocessing, image augmentation, balanced class weights, early stopping, learning-rate reduction, and model checkpoints.

## Project Structure

```text
Chest-XRay-API/
├── app/
│   ├── main.py              # FastAPI routes and inference pipeline
│   ├── model.py             # Loads archived Keras models
│   ├── preprocessing.py     # Image validation and DenseNet preprocessing
│   ├── gradcam.py           # Grad-CAM generation
│   └── schemas.py           # API response schemas
├── frontend/
│   ├── index.html           # Browser interface
│   ├── style.css
│   └── script.js
├── model/
│   ├── gatekeeper_best.zip  # X-ray vs non-X-ray model
│   ├── disease_best.zip     # Final disease-classification model
│   ├── disease_best_phase1.zip
│   └── model_config.json
├── evaluation/              # Saved evaluation and API screenshots
├── scripts/
│   └── diagnose_inference.py
├── tests/
├── chest-xray-classification.ipynb
├── requirements.txt
├── Dockerfile
└── compose.yaml
```

## Training Notebook

[`chest-xray-classification.ipynb`](chest-xray-classification.ipynb) performs the complete training workflow:

1. Loads the chest X-ray dataset and samples CIFAR-10 natural images.
2. Trains and evaluates the binary gatekeeper model.
3. Builds labels from the `COVID`, `NORMAL`, and `PNEUMONIA` directories.
4. Trains the three-class disease classifier in frozen and fine-tuning phases.
5. Compares both disease-model checkpoints and selects the better validation model.
6. Produces classification reports, a confusion matrix, training curves, and Grad-CAM visualizations.
7. Exports the model configuration used by the API.

Class mapping used by the trained model:

| Index | Class |
| --- | --- |
| `0` | `COVID` |
| `1` | `NORMAL` |
| `2` | `PNEUMONIA` |

## API Workflow

```text
Upload image
  -> validate file and decode image
  -> convert to RGB and resize to 224 × 224
  -> DenseNet preprocessing
  -> gatekeeper model
     -> not an X-ray: return rejection response
     -> chest X-ray: run disease classifier
        -> return class and probabilities
        -> optionally generate Grad-CAM heatmap
```

Supported formats are JPEG, PNG, BMP, and WEBP. Uploads are limited to 10 MiB; decoded images are limited to 25 million pixels.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the browser interface. |
| `GET` | `/health` | Reports whether both models are loaded. |
| `POST` | `/predict` | Runs the gatekeeper and disease classifier. |
| `POST` | `/gradcam` | Runs the pipeline and creates a Grad-CAM image for accepted X-rays. |
| `GET` | `/gradcam/image/{image_id}` | Returns a temporary generated Grad-CAM PNG. |

Use `image` as the multipart form field:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@/path/to/chest-xray.png"
```

Example response for an accepted X-ray:

```json
{
  "is_xray": true,
  "xray_confidence": 0.98,
  "message": null,
  "predicted_class": "PNEUMONIA",
  "confidence": 0.91,
  "probabilities": {
    "COVID": 0.03,
    "NORMAL": 0.06,
    "PNEUMONIA": 0.91
  }
}
```

For a non-X-ray upload, `is_xray` is `false`, disease fields are `null`, and no Grad-CAM image is created.

## Run Locally

Requirements: Python 3.11 and the model archives in the `model/` directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the following URLs after starting the server:

- Browser UI: `http://127.0.0.1:8000`
- Interactive API docs: `http://127.0.0.1:8000/docs`

## Docker

```bash
docker build -t chest-xray-api .
docker run --rm -p 8000:8000 chest-xray-api
```

Or start it with Docker Compose:

```bash
docker compose up --build
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite uses fake models, so the real TensorFlow model archives are not needed to run API tests.

To compare the API preprocessing and raw model predictions with the notebook-equivalent path:

```bash
python scripts/diagnose_inference.py /path/to/chest-xray.png
```

## Limitations

- The gatekeeper was trained with the notebook's chest X-ray data and CIFAR-10 images, so it may not generalize to every non-X-ray image.
- The disease classifier has not been externally or clinically validated.
- A prediction confidence score is not clinical certainty.
- Grad-CAM indicates regions that influenced the model output; it is not a medical explanation or diagnosis.

