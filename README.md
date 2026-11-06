# ML Workflows

Prefect 3 workflows for Sefaria's ML embedding pipeline. Runs on a self-hosted Prefect server with a Kubernetes work pool.

## Flows

### `create-dataset`
Pulls a raw data file from GCS, normalizes/chunks it, and writes the resulting dataset back to GCS.

### `train-embeddings`
Pulls the normalized dataset from GCS and fine-tunes a sentence embedding model. Runs on a GPU node.

### ModernBERT pretraining (`encoder-pretraining` tag)
Continues masked-language-model pretraining of ModernBERT on Sefaria's English corpus so the result can be fine-tuned for NER and other tasks. Run in this order (`extend-tokenizer` is optional):

| Flow | Runs on | What it does |
|---|---|---|
| `extend-tokenizer` | CPU | Adds Hebrew tokens to ModernBERT's tokenizer: vowel marks are stripped by the normalizer, and BPE merges are learned on the training split and prepended, so English tokenization is unchanged. Refuses to publish if special-token ids move or English tokenization changes. Writes `new_token_init.json` so new embeddings start as the mean of their old subtokens |
| `prepare-mlm-corpus` | CPU | Tokenizes the training JSONL (with the base or an extended tokenizer), drops documents that share a 10-word span with the NER gold set, packs 1024-token windows, writes `.npy` shards and a `manifest.json` to GCS, and reports how badly the tokenizer fragments inline Hebrew |
| `pretrain-modernbert` | A100 | Trains with the Hugging Face `Trainer`, saves a checkpoint to GCS every 10 minutes, resumes from the newest checkpoint on start, and writes the final model and a report |
| `pretrain-supervisor` | CPU | Calls `pretrain-modernbert` and starts it again when the run ends `Crashed` (spot eviction or unschedulable pod); optionally moves to on-demand A100s after repeated crashes |
| `evaluate-mlm` | A100 | Compares masked-token loss and accuracy of the pretrained model with base ModernBERT, per source, and prints fill-mask samples (base comparison is skipped when the vocabulary was extended) |
| `evaluate-ner-probe` | A100 | Fine-tunes each model on the `Sefaria/en_linker_gold_dataset` Person/Citation set with identical cross-validation splits and compares word-level span F1 against the first (baseline) model |

Start a run from the Prefect UI by running `pretrain-supervisor`. Use `gpu_parameters` to override training settings, for example `{"learning_rate": 0.0003, "max_tokens": 30000000}` for a learning-rate sweep run. Re-running with the same `run_name` resumes from that run's checkpoints, and a finished run is skipped.

GCS layout (bucket `sefaria-ml-models`):

```
modernbert/corpus/<data-version>/        # shards + manifest.json from prepare-mlm-corpus
modernbert/runs/<run_name>/checkpoints/  # last two checkpoints and latest.json
modernbert/runs/<run_name>/final/        # final model, written with a _SUCCESS marker last
modernbert/runs/<run_name>/training_report.json
```

The model is trained on web pages and library texts with mixed licenses and is for internal use only.

### Roshei tevot expansion (`roshei-tevot` tag)

Trains BEREL to pick the expansion of a roshei tevot abbreviation in context. The main model is siamese (context and expansion embedded separately, expansions precomputed); a cross-encoder trained on the same data measures how much accuracy that design gives up. Data is `Sefaria/berel-roshei-tevot` (private). The pods have no Hugging Face token, so the flows read a copy at `gs://sefaria-ml-models/roshei-tevot/data/berel-roshei-tevot/` (`dataset_repo` accepts either an HF dataset id or a `gs://` prefix; re-copy after the dataset changes). The dataset also holds `lexicon/surface_index.jsonl` (abbreviation to expansions, with library frequency) and `metadata/hebrew_titles.json` (English to Hebrew titles and categories, from the Sefaria table of contents). The lexicon, letter checker and spelling folds come from the `expand_roshei_tevot` experiment in `Sefaria/translation-engine`; the code is vendored in `app/roshei_tevot/steps/vendor/` (pinned commit in each file header; re-sync rather than edit).

| Flow | Runs on | What it does |
|---|---|---|
| `train-roshei-tevot` | A100 | `architecture: siamese` or `cross`. Hugging Face `Trainer` with GCS checkpoints every 10 minutes and resume on start, like `pretrain-modernbert`; run it through `pretrain-supervisor` with `deployment_name: train-roshei-tevot-flow/train-roshei-tevot` to restart after spot evictions. Logs validation top-1 at every evaluation |
| `build-expansion-index` | A100 | Embeds all ~57k lexicon expansions with a trained siamese model and writes `embeddings.npy` (fp16, under 100 MB) and `lexicon.json` to GCS. Load with `ExpansionIndex.load` and keep it in RAM |
| `evaluate-roshei-tevot` | A100 | Ranks each row's candidates on `val`, `test_seen_keys` and `test_unseen_keys` for each model, sweeps the frequency prior, picks it on val, and checks the success criterion. `gap_to_first` compares every model with the first one listed |

**Candidates.** For a token, the lexicon's expansions for the token as written and for each prefix-stripped reading, minus those whose letters cannot spell it (`checker.token_fits`). Training mines its hard negatives from this same lookup, and rows with fewer than two candidates are left out of training and evaluation (`min_candidates`). The spelling folds (`stem_key`, `fold_b`) decide when the gold expansion and a candidate are the same, so `ורבי יוסי` matches `רבי יוסי`.

**Input.** Siamese query: `[CLS] {Hebrew category} | {Hebrew title} [SEP] …before <abbr> {surface} </abbr> after… [SEP]`, pooled over the marked span. Siamese candidate: `[CLS] {expansion} [SEP]`, mean pooled. Cross-encoder: `[CLS] {expansion} [SEP] {category} | {title} [SEP] …context… [SEP]`, scored from CLS plus the marked span. Up to 100 words per side within 384 tokens, which keeps every val query whole. `<abbr>` and `</abbr>` are added tokens (BEREL has no unused slots).

**Loss.** Siamese: InfoNCE over the batch pool (targets plus up to 7 mined negatives per row, deduplicated by folded spelling), plus `within_weight` times cross-entropy over each row's own candidates. Cross-encoder: cross-entropy over each row's target and up to 7 mined negatives. Rows are resampled with weight `1 / count(key)^0.5`; synthetic swapped-back rows count half; gold and LLM labels count the same.

**Scoring.** `score = model score + λ · log(1 + lexicon frequency)`, with λ swept over `prior_weights` and chosen on val. Success means top-1 at least 3 points above the most-frequent-expansion baseline on `val` and `test_seen_keys`. At inference, `decide(ranked, min_score, min_margin)` abstains when the answer is weak or too close to the runner-up.

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

The ML cluster is `cluster-1` (`us-east1-b`) in the `development-205018` project. `pretrain-modernbert` and `evaluate-mlm` select nodes with the labels `cloud.google.com/gke-accelerator: nvidia-tesla-a100` and `cloud.google.com/gke-provisioning: spot`. The existing `gpu-containernodepool` has P100s, so the selector keeps jobs off it. Create the pool once (A100 40GB is only offered in `us-east1-b`):

```bash
gcloud container node-pools create a100-spot \
  --project development-205018 --cluster cluster-1 --zone us-east1-b \
  --machine-type a2-highgpu-1g --spot \
  --accelerator type=nvidia-tesla-a100,count=1,gpu-driver-version=latest \
  --enable-autoscaling --min-nodes 0 --max-nodes 1 \
  --disk-size 200 --disk-type pd-ssd
```

Quota in `development-205018` / `us-east1`: `PREEMPTIBLE_NVIDIA_A100_GPUS` is 1 (one spot A100 at a time, so no parallel sweep runs) and `NVIDIA_A100_GPUS` (on-demand) is 0. The supervisor's on-demand fallback is therefore off (`allow_on_demand: false`). To enable it, request on-demand A100 quota, create an `a100-ondemand` pool the same way without `--spot`, and set `allow_on_demand: true`.

The work pool's base job template is not deployed by CI. After changing `prefect_work_pools/sefaria-k8s-base-job-template.json` (this change adds a `node_selector` variable), update the `sefaria-k8s` pool with it before deploying.

The `ml-workflows-runner` Kubernetes service account maps to `ml-workflows-sa@development-205018.iam.gserviceaccount.com`. That account needs read and write access to the `sefaria-ml-models` bucket, which it does not have by default:

```bash
gcloud storage buckets add-iam-policy-binding gs://sefaria-ml-models \
  --member serviceAccount:ml-workflows-sa@development-205018.iam.gserviceaccount.com \
  --role roles/storage.objectAdmin
```

## Docker image

A single image is used for the embedding flows, based on `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`. The ModernBERT GPU flows use a second image, `Dockerfile.gpu-train`, based on torch 2.6 (the minimum transformers 4.57 needs to resume a Trainer checkpoint) with a prebuilt flash-attn wheel. Both are built and pushed to Google Artifact Registry on every merge to `main`.

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
