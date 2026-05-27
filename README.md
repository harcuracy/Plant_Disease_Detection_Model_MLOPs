# End-to-End Plant Disease Detection MLOps Project

This project is an end-to-end deep learning workflow for detecting plant leaf diseases from images. It includes Kaggle data ingestion, balanced subset creation, transfer learning with MobileNetV2, model training, evaluation, a FastAPI prediction service, Docker packaging, and GitHub Actions CI/CD to AWS ECR.

## Features

- Kaggle dataset download using `kagglehub`
- Balanced subset creation for faster local experimentation
- Transfer learning with TensorFlow/Keras MobileNetV2
- Config-driven pipeline using `config/config.yaml` and `params.yaml`
- DVC pipeline definition for reproducible stages
- FastAPI web app and API for image upload and prediction
- Docker image for deployment
- GitHub Actions CI/CD pipeline with AWS ECR image push
- Centralized logging to console and `logs/running_logs.log`
- Smoke tests for config, API health, and notebook validity
- Research notebook in `research/trials.ipynb`

## Project Structure

```text
.
|-- app.py
|-- config/
|   `-- config.yaml
|-- dvc.yaml
|-- Dockerfile
|-- main.py
|-- params.yaml
|-- pyproject.toml
|-- research/
|   `-- trials.ipynb
|-- src/
|   `-- cnnClassifier/
|       |-- components/
|       |-- config/
|       |-- constants/
|       |-- entity/
|       |-- pipeline/
|       |-- utils/
|       `-- logger.py
|-- templates/
|   `-- index.html
`-- tests/
```

## Dataset

The default dataset is configured in `config/config.yaml`:

```yaml
data_ingestion:
  source_URL: vipoooool/new-plant-diseases-dataset
```

The dataset is large, so this project creates a smaller balanced subset. Control the size in `params.yaml`:

```yaml
data_ingestion:
  validation_split: 0.2
  max_images_per_class: 100
  random_seed: 42
```

Increase `max_images_per_class` when you are ready for longer training.

## Setup

Create and activate a virtual environment:

```powershell
uv venv --python 3.11
.venv\Scripts\activate
```

Install dependencies:

```powershell
uv sync
```

Copy the sample environment file if you want local runtime variables:

```powershell
copy .env.example .env
```

## Run the Pipeline

Run all stages:

```powershell
python main.py
```

Run individual stages:

```powershell
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
python src/cnnClassifier/pipeline/stage_03_training.py
python src/cnnClassifier/pipeline/stage_04_evaluation.py
```

Using DVC:

```powershell
dvc repro
```

Pipeline outputs are written under `artifacts/`.

Inspect the DVC pipeline graph:

```powershell
dvc dag
```

Show evaluation metrics:

```powershell
dvc metrics show
```

Current metrics:

```text
accuracy: 0.84605
loss: 0.55516
```

## Training Configuration

Model and training settings live in `params.yaml`:

```yaml
prepare_base_model:
  image_size: [224, 224, 3]
  learning_rate: 0.0001
  include_top: false
  weights: imagenet
  classes: 38
  dropout_rate: 0.2

training:
  batch_size: 16
  epochs: 20
  augmentation: true
```

The class count is automatically resolved from the subset training folder when available.

## Run the FastAPI App

After training creates `artifacts/training/model.h5` and `artifacts/training/class_names.yaml`, start the API:

```powershell
python app.py
```

Open the upload UI:

```text
http://localhost:8080
```

Open interactive API docs:

```text
http://localhost:8080/docs
```

Health check:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Upload the image using the form field name `image`.

Prediction logic lives in `src/cnnClassifier/components/prediction.py`, while `app.py` handles API routes and templates.

## Tests

Run smoke tests:

```powershell
pytest -q
```

The tests check configuration loading, FastAPI health response, and notebook JSON validity without running TensorFlow training.

## Docker

Build the image:

```powershell
docker build -t plant-disease-detection:latest .
```

Run the container:

```powershell
docker run -p 8080:8080 plant-disease-detection:latest
```

The app expects these files inside the container or mounted at runtime:

```text
artifacts/training/model.h5
artifacts/training/class_names.yaml
```

Override model paths:

```powershell
docker run -p 8080:8080 `
  -e MODEL_PATH=/app/artifacts/training/model.h5 `
  -e CLASS_NAMES_PATH=/app/artifacts/training/class_names.yaml `
  plant-disease-detection:latest
```

## CI/CD with GitHub Actions and AWS ECR

The workflow is defined in `.github/workflows/ecr.yml`.

It performs:

- Python compile validation
- Smoke tests with `pytest`
- YAML validation
- Notebook JSON validation
- Docker build validation on pull requests
- Docker build and push to AWS ECR on `main` pushes and manual workflow runs

Add these GitHub repository secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
```

Default ECR repository:

```text
plant-disease-detection
```

Images are pushed as:

```text
<aws_account_id>.dkr.ecr.<region>.amazonaws.com/plant-disease-detection:<git_sha>
<aws_account_id>.dkr.ecr.<region>.amazonaws.com/plant-disease-detection:latest
```

## Notebook Workflow

Use `research/trials.ipynb` as a normal research notebook:

1. Load libraries and config
2. Download the Kaggle dataset
3. Inspect image folders
4. Create a small balanced subset
5. Visualize images
6. Build TensorFlow datasets
7. Train a MobileNetV2 transfer-learning model
8. Plot metrics and test one prediction

## Logs

Logs are written to:

```text
logs/running_logs.log
```

The same logs are also printed to the console.

## Production Notes

- Start with a small `max_images_per_class` and low `epochs` value while validating the workflow.
- Keep raw data and large generated models out of Git.
- For production deployment, store trained model artifacts in S3, EFS, or a model registry, then mount or download them at container startup.
