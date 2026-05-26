# ML Workflows

Prefect 3 workflows for Sefaria's ML embedding pipeline. Runs on a self-hosted Prefect server with a Kubernetes work pool.

## Flows

### `create-dataset`
Pulls a raw data file from GCS, normalizes/chunks it, and writes the resulting dataset back to GCS.

### `train-embeddings`
Pulls the normalized dataset from GCS and fine-tunes a sentence embedding model. Runs on a GPU node.

## Repository structure

```
app/
  embeddings/
    flows/
      create_dataset.py     # create-dataset flow
      train_embeddings.py   # train-embeddings flow
    steps/
      patot/                # PATOT chunker port and JSON-backed segment loader
  utils/
    gcs.py                  # shared GCS helpers
chart/                      # Helm chart for the Prefect Kubernetes worker
Dockerfile                  # pytorch/cuda base image shared by both flows
prefect.yaml                # Prefect deployment definitions
requirements.txt
```

## Infrastructure

Both flows use the `sefaria-k8s` Kubernetes work pool. The GPU flow adds a `nvidia.com/gpu` toleration and resource limit so it's scheduled on a GPU node.

The Prefect worker runs in the `ml-infra` namespace. Flow jobs are also created in `ml-infra` via `job_variables.namespace`.

## Docker image

A single image is used for both flows, based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`. It is built and pushed to Google Artifact Registry on every merge to `main`.

Image path: `us-east1-docker.pkg.dev/<DEV_GKE_PROJECT>/containers/mlworkflows`

## GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| `build.yaml` | Push to `main` | Builds and pushes the Docker image, then deploys flows |
| `deploy-flows.yaml` | Manual or called by `build.yaml` | Runs `prefect deploy --all` |

### Required secrets

| Secret | Description |
|---|---|
| `DEV_GKE_PROJECT_ID` | Numeric GCP project ID (for Workload Identity) |
| `DEV_GKE_PROJECT` | GCP project name (for image registry path) |
| `DEV_GKE_SA` | GCP service account for Workload Identity auth |
| `PREFECT_API_URL` | URL of the self-hosted Prefect server API |

## Deploying the worker

```bash
helm upgrade --install prefect-worker chart/ --namespace ml-infra
kubectl rollout restart deployment/dev-prefect-worker -n ml-infra
```

## Local development

Copy `.env.example` to `.env` and fill in values, then run flows directly:

```bash
pip install -r requirements.txt
python -c "from app.embeddings.flows.create_dataset import create_dataset_flow; create_dataset_flow(...)"
```
