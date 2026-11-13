# Predictive Maintenance API on Google Cloud Run

A machine failure prediction service trained on industrial sensor data, served with FastAPI, containerized with Docker, and deployed serverless on Google Cloud Run.

**Live API docs:** https://model-api-633970234530.us-east1.run.app/docs

Built for the Cloud Computing lab of the MLOps course at Northeastern University (labs based on the course's GCP_Labs / Cloud_Runner_Labs). It builds on my earlier labs in the same course: CI/CD with GitHub Actions, Docker, and FastAPI.

## Architecture

| Component | Choice |
|---|---|
| Model | scikit-learn RandomForestClassifier |
| Serving | FastAPI + Uvicorn, Pydantic input validation |
| Container | python:3.13-slim |
| Registry | Google Artifact Registry (us-east1) |
| Hosting | Google Cloud Run (serverless, scales to zero) |

## Changes from the original lab

- **New dataset and problem:** binary machine failure prediction on the AI4I 2020 Predictive Maintenance dataset, instead of the lab's sample model.
- **Physics-based feature engineering** that measurably improved the model (see Results).
- **Shared feature module** used by both training and serving to prevent training-serving skew.
- **Production-style API:** health check, model info endpoint with live metrics, validated inputs, model loaded once at startup.
- **Tested image deployed as-is:** the image was built and verified locally, then pushed unchanged to Artifact Registry.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI Machine Learning Repository): 10,000 synthetic records of milling machine sensor readings with a binary `Machine failure` label.

Inputs: product type (L/M/H), air temperature, process temperature, rotational speed, torque, and tool wear.

Failures are rare (about 3.4% of rows), so accuracy is misleading. A model that always predicts "no failure" scores about 97% accuracy. This project reports **recall and F1 on the failure class** instead, and trains with `class_weight="balanced"`.

The individual failure mode columns (TWF, HDF, PWF, OSF, RNF) are **excluded from training** because they directly encode the label and would leak it.

## Feature engineering

The dataset's failure labels are generated from physical rules, such as overstrain when torque times tool wear exceeds a threshold, or power failure when mechanical power leaves a safe range. A random forest working on raw columns has to approximate these products with many axis-aligned splits, which makes it unreliable in sparse regions of the data.

Three engineered features encode those mechanisms directly:

| Feature | Definition | Failure mechanism |
|---|---|---|
| `power_w` | torque x rotational speed x 2 x pi / 60 | Power failure |
| `strain` | torque x tool wear | Overstrain failure |
| `temp_diff_k` | process temperature - air temperature | Heat dissipation failure |

These are computed in `app/features.py`, which both `train.py` and the API import. Serving can never compute features differently from training.

## Results

Test set: 2,000 rows (stratified 80/20 split), 68 failures.

| Model | Precision (failure) | Recall (failure) | F1 (failure) |
|---|---|---|---|
| Raw features only | 0.77 | 0.72 | 0.74 |
| Raw + engineered features | **0.95** | **0.82** | **0.88** |

**Model probing.** Beyond aggregate metrics, I tested the model on a high-stress input (type L, torque 65 Nm, tool wear 230 min, 1551 rpm) that should fail under the dataset's overstrain and power rules:

| Model | Failure probability | Prediction |
|---|---|---|
| Raw features only | 0.485 | No failure (wrong) |
| Raw + engineered features | 0.935 | Failure (correct) |

The baseline model's aggregate metrics looked reasonable, but probing exposed this miss. The engineered features fixed it.

**Deployment verification.** The same request returns identical predictions from the local Uvicorn server, the local Docker container, and the live Cloud Run service.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check, confirms the model is loaded |
| GET | `/model-info` | Model type, dataset, features, and test metrics |
| POST | `/predict` | Returns failure prediction and probability |
| GET | `/docs` | Interactive Swagger UI |

Example request:

```json
{
  "type": "L",
  "air_temperature_k": 298.1,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1551,
  "torque_nm": 65,
  "tool_wear_min": 230
}
```

Example response:

```json
{
  "failure": true,
  "failure_probability": 0.935
}
```

Invalid inputs, such as negative RPM or an unknown product type, return a 422 validation error instead of reaching the model.

## Project structure

The `data/` folder is not committed. Download the dataset as described below.

## Run locally

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

Download the data into `data/ai4i2020.csv` from the UCI page linked above, then:

```bash
python train.py
uvicorn app.main:app --port 8080
```

Open http://localhost:8080/docs.

## Run with Docker

```bash
docker build -t model-api .
docker run --rm -p 8080:8080 model-api
```

## Deploy to Cloud Run

One-time setup:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create mlops-repo --repository-format=docker --location=us-east1
gcloud auth configure-docker us-east1-docker.pkg.dev
```

Push the tested image and deploy:

```bash
docker tag model-api us-east1-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/model-api:v1
docker push us-east1-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/model-api:v1

gcloud run deploy model-api --image us-east1-docker.pkg.dev/YOUR_PROJECT_ID/mlops-repo/model-api:v1 --region us-east1 --allow-unauthenticated --memory 1Gi --max-instances 2
```

## Design decisions

- **Model loaded once at startup** through FastAPI's `lifespan` handler, not per request. On Cloud Run, cold starts are already the main latency cost.
- **Port read from `$PORT`**, which Cloud Run sets at runtime.
- **Shell-form `CMD` with `exec`.** Shell form is needed so `${PORT}` expands. `exec` replaces the shell with Uvicorn so shutdown signals reach the app. This is why Docker's `JSONArgsRecommended` warning is safe to ignore here.
- **Python version pinned to match training (3.13).** Pickled scikit-learn models are not guaranteed to load across library or Python versions, so the container mirrors the training environment exactly via `pip freeze`.
- **Cost controls.** `--max-instances 2` caps scaling. Cloud Run scales to zero when idle, so the service costs nothing between requests. The first request after idle time may take a few seconds (cold start).

## Limitations

- **Some failures are unpredictable by design.** AI4I includes random failures with no sensor signal and tool wear failures assigned randomly within a wear window. No model can fully catch these, which limits achievable recall.
- **Probabilities are not calibrated.** Random forest probabilities are the share of trees voting "failure," not true probabilities. Calibration would be needed before using them for risk decisions.
- **Default 0.5 threshold.** For maintenance use, missing a failure usually costs more than a false alarm, so the threshold should be tuned on a validation set.
- **Model baked into the image.** Updating the model requires a rebuild and redeploy. A model registry or Cloud Storage would decouple model versions from code.

## Next steps

- GitHub Actions workflow to build and deploy on every push to `main`, using Workload Identity Federation instead of service account keys.
- Deploy the same image to Google Kubernetes Engine and compare it with Cloud Run on scaling, cost, and operational effort.

## Author

Jairaghavendra Sridhar, MS Data Analytics Engineering, Northeastern University
[GitHub](https://github.com/jaira26) | [LinkedIn](https://www.linkedin.com/in/jairaghavendrasridhar26)
