# ML Workflows

Prefect 3 workflows for Sefaria's ML embedding pipeline. Runs on a self-hosted Prefect server with a Kubernetes work pool.

## Flows

### `create-dataset`
Pulls a raw data file from GCS, normalizes/chunks it, and writes the resulting dataset back to GCS.

### `train-embeddings`
Pulls the normalized dataset from GCS and fine-tunes a sentence embedding model. Runs on a GPU node.

### ModernBERT pretraining (`encoder-pretraining` tag)
Continues masked-language-model pretraining of ModernBERT on Sefaria's English corpus so the result can be fine-tuned for NER and other tasks. Four flows, run in this order:

| Flow | Runs on | What it does |
|---|---|---|
| `prepare-mlm-corpus` | CPU | Tokenizes the training JSONL, packs it into 1024-token windows, writes `.npy` shards and a `manifest.json` to GCS, and reports how badly the tokenizer fragments inline Hebrew |
| `pretrain-modernbert` | A100 | Trains with the Hugging Face `Trainer`, saves a checkpoint to GCS every 10 minutes, resumes from the newest checkpoint on start, and writes the final model and a report |
| `pretrain-supervisor` | CPU | Calls `pretrain-modernbert` and starts it again when the run ends `Crashed` (spot eviction or unschedulable pod), moving to on-demand A100s after repeated crashes |
| `evaluate-mlm` | A100 | Compares masked-token loss and accuracy of the pretrained model with base ModernBERT, per source, and prints fill-mask samples |

Start a run from the Prefect UI by running `pretrain-supervisor`. Use `gpu_parameters` to override training settings, for example `{"learning_rate": 0.0003, "max_tokens": 30000000}` for a learning-rate sweep run. Re-running with the same `run_name` resumes from that run's checkpoints, and a finished run is skipped.

GCS layout (bucket `sefaria-ml-models`):

```
modernbert/corpus/<data-version>/        # shards + manifest.json from prepare-mlm-corpus
modernbert/runs/<run_name>/checkpoints/  # last two checkpoints and latest.json
modernbert/runs/<run_name>/final/        # final model, written with a _SUCCESS marker last
modernbert/runs/<run_name>/training_report.json
```

The model is trained on web pages and library texts with mixed licenses and is for internal use only.

## Repository structure

```
app/
  embeddings/
    flows/
      create_dataset.py     # create-dataset flow
      train_embeddings.py   # train-embeddings flow
  encoder_pretraining/
    flows/
      prepare_mlm_corpus.py     # prepare-mlm-corpus flow
      pretrain_modernbert.py    # pretrain-modernbert flow (GPU)
      pretrain_supervisor.py    # pretrain-supervisor flow (restarts after preemption)
      evaluate_mlm.py           # evaluate-mlm flow
    steps/
      data/                     # tokenizing, packing, masking, shard dataset
      training/                 # Trainer pipeline, GCS checkpoint callback, model loading
      evaluation/               # masked-token metrics, fill-mask samples
  utils/
    gcs.py                  # shared GCS helpers
chart/                      # Helm chart for the Prefect Kubernetes worker
Dockerfile                  # pytorch/cuda base image shared by both flows
Dockerfile.gpu-train        # newer torch + flash-attn for ModernBERT pretraining
prefect.yaml                # Prefect deployment definitions
requirements.txt
```

## Infrastructure

Both flows use the `sefaria-k8s` Kubernetes work pool. The GPU flow adds a `nvidia.com/gpu` toleration and resource limit so it's scheduled on a GPU node.

The Prefect worker runs in the `ml-infra` namespace. Flow jobs are also created in `ml-infra` via `job_variables.namespace`.

### A100 spot node pool for ModernBERT pretraining

`pretrain-modernbert` and `evaluate-mlm` select nodes with the labels `cloud.google.com/gke-accelerator: nvidia-tesla-a100` and `cloud.google.com/gke-provisioning: spot` (the supervisor switches to `standard` for on-demand). The existing `gpu-containernodepool` has P100s, so the selector keeps jobs off it. Create the pools once (A100 40GB is only offered in `us-east1-b`):

```bash
gcloud container node-pools create a100-spot \
  --cluster cluster-1 --zone us-east1-b \
  --machine-type a2-highgpu-1g --spot \
  --accelerator type=nvidia-tesla-a100,count=1,gpu-driver-version=latest \
  --enable-autoscaling --min-nodes 0 --max-nodes 1 \
  --disk-size 200 --disk-type pd-ssd

# Same again with the name a100-ondemand and no --spot, for the supervisor's fallback.
```

The work pool's base job template is not deployed by CI. After changing `prefect_work_pools/sefaria-k8s-base-job-template.json` (this branch adds a `node_selector` variable), update the `sefaria-k8s` pool with it before deploying. The `ml-workflows-runner` service account also needs read and write access to `sefaria-ml-models`.

## Docker image

A single image is used for the embedding flows, based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`. The ModernBERT GPU flows use a second image, `Dockerfile.gpu-train`, based on torch 2.5.1 with a prebuilt flash-attn wheel. Both are built and pushed to Google Artifact Registry on every merge to `main`.

Image paths: `us-east1-docker.pkg.dev/<DEV_GKE_PROJECT>/containers/mlworkflows` and `.../containers/mlworkflows-gpu-train`

## GitHub Actions

| Workflow | Trigger | What it does |
|---|---|---|
| `build.yaml` | Push to `main` | Builds and pushes both Docker images, then deploys flows |
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
