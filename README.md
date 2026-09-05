# DistillFW — Managed Distillation Framework on Google Cloud Platform

DistillFW is an end-to-end, production-grade platform on Google Cloud Platform (GCP)—consisting of a **FastAPI backend**, a **React/Vite Web UI**, and an **orchestration engine**—to manage the complete lifecycle of distilling knowledge from a proprietary **Teacher Model** (Gemini 2.5 Pro / Flash via Vertex AI) into a compact, parameter-efficient open-source **Student Model** (Gemma 2, Llama 3, Mistral) tailored to specific tasks using PEFT QLoRA.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                              Client Web Browser                                   |
|               React / Vite / Tailwind UI with Real-time Telemetry                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                    Apigee API Gateway / Cloud Run Reverse Proxy                   |
|                   Route: /api/* -> Backend  |  /* -> Web UI                       |
+--------------------+------------------------------------+-------------------------+
                     |                                    |
                     v                                    v
+--------------------+-------------------+   +------------+-------------------------+
|           FastAPI Backend Service      |   |        Google Cloud Storage          |
|    - Deterministic Status Inference    |<->|       gs://<bucket>/<project-id>/    |
|    - Dataset Validation & Auto-Split   |   |   ├── config.yaml                    |
|    - Gemini Teacher Inference & CoT    |   |   ├── history.json                   |
|    - Hardware Calibration Probe        |   |   ├── data/ (input, split, teacher)  |
|    - Vertex CustomJob Orchestration    |   |   ├── cost/ (cost_estimate.json)     |
|    - 3-Tier Multi-Metric Evaluation    |   |   ├── training/ (metrics, heartbeat) |
|    - Vertex AI vLLM Endpoint Deploy    |   |   ├── evaluation/ (results, preds)   |
+--------------------+-------------------+   |   └── deployment/ (endpoint_metadata)|
                     |                       +--------------------------------------+
                     v                                    ^
+--------------------+-------------------+                |
|       Vertex AI Training / Prediction   |                |
|  - CustomJob: DistillationTrainer       |----------------+
|  - Endpoint: vLLM High-Throughput Engine|
+-----------------------------------------+
```

---

## Supported Distillation Methods

1. **Method 1: Sequence-Level KD (SeqKD)** *(Off-policy default)*: Standard cross-entropy over teacher completions.
2. **Method 2: Distilling Step-by-Step (CoT Reasoning Distillation)**: Multi-task weighted loss combining thinking rationale traces ($\mathcal{L}_{\text{think}}$) and final answers ($\mathcal{L}_{\text{resp}}$).
3. **Method 3: Generalized Knowledge Distillation (GKD) & On-Policy Teacher-as-a-Judge**: Policy updates on student rollouts using teacher feedback / rewards.
4. **Method 4: Top-$k$ Soft Target KD**: Top-5 soft target KL divergence with temperature scaling for shared vocabularies.

---

## Project Structure

```
distillfw/
├── backend/                  # FastAPI orchestration backend
│   ├── api/routes/           # Workspaces, config, dataset, teacher, cost, training, eval, deployment, logs
│   ├── core/                 # Models, settings, Pydantic schemas
│   ├── services/             # GCS storage, dataset, teacher, probe, training, eval, deployment, logger
│   └── main.py               # FastAPI entrypoint & SPA static asset server
├── frontend/                 # React, Vite, Tailwind CSS single page application
│   ├── src/components/       # Header, ConfigForm, DatasetTab, TeacherTab, CostTab, TrainingTab, etc.
│   └── src/api.js            # API client
├── trainer/                  # PyTorch custom trainer container for Vertex AI CustomJob
│   ├── distillation_loss.py  # Loss functions for all 4 distillation methods
│   ├── callbacks.py          # GCSProgressCallback streaming metrics.jsonl & heartbeat.json
│   └── train.py              # Main training script subclassing transformers.Trainer
├── terraform/                # Infrastructure as Code
│   ├── modules/storage/      # GCS bucket with CORS policies
│   ├── modules/artifact_reg/ # Private Docker container registry
│   ├── modules/cloud_run/    # Backend & Frontend Cloud Run services
│   ├── modules/apigee/       # Apigee Gateway & routing
│   └── modules/iam/          # Service accounts & least-privilege roles
├── tests/                    # Comprehensive unit and integration test suite
├── deploy.sh                 # Pre-flight checks, provisioning, sample data seeding, and --reset
└── examples/                 # Sample configurations and datasets
    ├── sample_config.yaml    # Master configuration specification
    └── sample_dataset.jsonl  # 100 math problems with numeric responses
```

---

## Quickstart & Local Execution

### 1. Run Automated Setup & Deploy
```bash
./deploy.sh --dry-run
```
This performs:
- Dependency validation (`gcloud`, `terraform`, `docker`, `python3`, `node`, `npm`).
- Compiles the frontend SPA into `frontend/dist/`.
- Initializes Terraform modules and providers.
- Creates workspace bucket `distillfw-workspaces` and initializes `distill-gemma-math-v1` in `DATASET_READY` status with 100 math problems split 80/10/10.

### 2. Start the Backend & Web UI
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```
Open your browser at **http://localhost:8080**:
- **Main Bucket Combobox**: Defaults to `distillfw-workspaces`.
- **Workspace Folder Selector**: Selects `distill-gemma-math-v1`.
- **Status Badge**: Automatically derives and displays `DATASET_READY`.
- **Configuration Form**: Full interactive controls for all `config.yaml` parameters.
- **Collapsible Bottom Panel**: Expand to view live telemetry and execution logs.

### 3. Run Test Suite
```bash
python3 -m pytest tests/
```

### 4. Teardown / Reset
```bash
./deploy.sh --reset
```
Destroys all created GCP Terraform resources and cleans up workspace artifacts.

---

## Documentation

- **[Installation Guide](docs/install.md)**: Detailed GCP setup, required IAM roles, APIs, Terraform provisioning, Docker images, and local development.
- **[User Guide](docs/userguide.md)**: Step-by-step example workflow for distilling Gemini 2.5 Pro reasoning into Gemma 2 9B, covering all 7 stages from both the Web UI and REST API.

