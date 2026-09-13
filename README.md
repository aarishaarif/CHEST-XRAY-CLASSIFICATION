# Chest X-Ray Classification API

A deep learning based Chest X-Ray Classification system using **DenseNet121**, **FastAPI**, **Docker**, and **Grad-CAM**.

The system uses a two-stage classification pipeline:

1. **Gatekeeper Model** — determines whether the uploaded image is a valid chest X-ray.
2. **Disease Classification Model** — classifies valid chest X-rays into:

   * COVID
   * NORMAL
   * PNEUMONIA

---

## Features

* DenseNet121-based image classification
* Two-stage prediction pipeline
* FastAPI REST API
* Dockerized application
* Web-based frontend
* Image validation
* Confidence scores and class probabilities
* Grad-CAM visual explanations
* Automated API tests
* Health-check endpoint

---

## Model Architecture

### Stage 1 — X-Ray Gatekeeper

The first model checks whether the uploaded image is a valid chest X-ray.

If the confidence is below the configured threshold, the image is rejected and is not passed to the disease classifier.

```text
Uploaded Image
      ↓
Gatekeeper Model
      ↓
Valid Chest X-Ray?
   ↙          ↘
 No            Yes
 ↓              ↓
Reject       Disease Model
                  ↓
          COVID / NORMAL / PNEUMONIA
```

The gatekeeper threshold is:

```text
0.5
```

### Stage 2 — Disease Classification

If the image passes the gatekeeper, the disease classification model predicts one of three classes:

```text
COVID
NORMAL
PNEUMONIA
```

---

## Project Structure

```text
Chest-XRay API/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── evaluation/
│   ├── Confusion matrix.png
│   ├── Precision and recall.png
│   ├── gradcam pic.png
│   ├── gradcam response.png
│   ├── Post,predict.png
│   ├── post,gradcam.png
│   └── predict response.png
│
├── fastapi-docker/
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── schemas.py
│   │   └── gradcam.py
│   │
│   ├── frontend/
│   │   ├── index.html
│   │   ├── script.js
│   │   └── style.css
│   │
│   ├── model/
│   │   ├── disease_best.zip
│   │   ├── disease_best_phase1.zip
│   │   ├── gatekeeper_best.zip
│   │   └── model_config.json
│   │
│   ├── scripts/
│   │   └── diagnose_inference.py
│   │
│   ├── tests/
│   │   └── test_api.py
│   │
│   ├── Dockerfile
│   ├── compose.yaml
│   ├── requirements.txt
│   └── requirements-dev.txt
│
├── chest-xray-classification.ipynb
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## FastAPI Application

The API is implemented inside:

```text
fastapi-docker/app/
``
```
