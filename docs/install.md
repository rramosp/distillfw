# DistillFW — GCP Installation & Deployment Guide

This guide provides step-by-step instructions for installing, configuring, and deploying DistillFW on Google Cloud Platform (GCP).

---

## 1. Prerequisites & Environment Setup

### 1.1. Required CLI Tools
Ensure the following tools are installed on your workstation or Cloud Shell environment:

| Dependency | Minimum Version | Installation Command / Link |
| :--- | :--- | :--- |
| **Google Cloud SDK (`gcloud`)** | 450.0.0+ | [Install gcloud CLI](https://cloud.google.com/sdk/docs/install) |
| **Terraform** | 0.13+ (1.5+ recommended) | [Install Terraform](https://developer.hashicorp.com/terraform/install) |
| **Docker** | 24.0+ | [Install Docker Engine](https://docs.docker.com/engine/install/) |
| **Python** | 3.11+ | Python 3.11 or 3.12 |
| **Node.js & npm** | Node v18+ / npm 9+ | [Install Node.js](https://nodejs.org/) |

### 1.2. Google Cloud Authentication & Project Setup
Login to Google Cloud and designate your target project:

```bash
# Authenticate gcloud user credentials
gcloud auth login

# Set application default credentials for client libraries
gcloud auth application-default login

# Set active project
export GCP_PROJECT_ID="your-gcp-project-id"
gcloud config set project "$GCP_PROJECT_ID"
```

### 1.3. Verify Active Billing
Vertex AI Custom Training and Cloud Run require an active billing account linked to your project:

```bash
gcloud beta billing projects describe "$GCP_PROJECT_ID" --format="value(billingEnabled)"
# Must output: True
```

### 1.4. Required IAM Roles & Permissions
The deploying user or service account must possess the following roles on the target project (or `roles/owner`):

- `roles/aiplatform.admin` (Vertex AI Custom Training, Endpoints, Model Registry)
- `roles/storage.admin` (Google Cloud Storage workspace management)
- `roles/run.admin` (Cloud Run backend and frontend service deployment)
- `roles/apigee.admin` (Apigee API Gateway proxy routing)
- `roles/artifactregistry.admin` (Docker image registry management)
- `roles/iam.serviceAccountUser` (ActAs permissions for service accounts)

> [!NOTE]
> `deploy.sh` automatically audits your current IAM roles. If any required roles are missing, it checks whether your account has administrative authority (`roles/resourcemanager.projectIamAdmin` or `roles/owner`) to self-grant them, prompts for your confirmation, and adds the bindings. If your account lacks permission to grant roles, `deploy.sh` outputs a formatted message with exact justification details and ready-to-run `gcloud` commands to send to your GCP Project Administrator.

---

## 2. Automated One-Click Deployment (`deploy.sh`)

DistillFW includes an automated deployment script ([`deploy.sh`](file:///usr/local/google/home/raulramos/projects/distillfw/deploy.sh)) that orchestrates pre-flight checks, API enablement, frontend compilation, Terraform infrastructure provisioning, and sample data seeding.

### 2.1. Deploy to Google Cloud Platform
To run the full end-to-end deployment against GCP:

```bash
./deploy.sh

# Or with automatic confirmation of prompts:
./deploy.sh --yes
```

### 2.2. What `deploy.sh` Performs
1. **Pre-flight Checks & IAM Verification**:
   - Confirms local dependencies: `gcloud`, `terraform`, `docker`, `python3`, `node`, `npm`.
   - Validates authenticated identity and active billing account.
   - Evaluates project IAM policy for missing roles.
   - If missing roles are found and you have authority (`roles/resourcemanager.projectIamAdmin` / `roles/owner`), prompts for confirmation and self-grants them.
   - If missing roles cannot be self-granted, halts and generates a detailed copy-pasteable request to send to your GCP Administrator.
2. **Google Cloud APIs Enablement**:
   Enables all required APIs:
   - `aiplatform.googleapis.com`
   - `run.googleapis.com`
   - `apigee.googleapis.com`
   - `storage.googleapis.com`
   - `artifactregistry.googleapis.com`
   - `cloudbuild.googleapis.com`
   - `iam.googleapis.com`
3. **Frontend Compilation**:
   Builds the React/Vite/Tailwind SPA into `frontend/dist/`.
4. **Terraform Provisioning**:
   - Creates the GCS bucket (`distillfw-workspaces`) with uniform access and CORS policies.
   - Sets up Artifact Registry repository `distillfw-docker-repo`.
   - Provisions least-privilege service accounts (`distillfw-backend-sa`, `distillfw-trainer-sa`).
   - Deploys Cloud Run services and Apigee routes.
5. **Sample Project Seeding (Section 8)**:
   - Initializes project `distill-gemma-math-v1` in `distillfw-workspaces`.
   - Populates `config.yaml` from `sample_config.yaml`.
   - Validates and splits `sample_dataset.jsonl` (100 numeric math problems) into 80% train, 10% val, 10% test.
   - Records initialization in `history.json`.
   - Confirms the project state is in **`DATASET_READY`**.

### 2.3. Dry-Run & Local Verification Mode
If you wish to test pre-flight checks, compile the frontend, initialize Terraform modules, and seed a local/offline workspace without invoking billable GCP calls:

```bash
./deploy.sh --dry-run
```

### 2.4. Teardown / Complete Reset Mode (`--reset`)
To remove and completely tear down all GCP cloud resources and local artifacts created by DistillFW:

```bash
./deploy.sh --reset

# Or non-interactively with auto-confirmation:
./deploy.sh --reset --yes
```

`--reset` executes an exhaustive 7-step cleanup process:
1. **Terraform Destroy**: Destroys all state-tracked resources via `terraform destroy -auto-approve`.
2. **Cloud Run Services**: Discovers and deletes all DistillFW Cloud Run services (`distillfw-backend`, `distillfw-frontend`, and any service matching `distillfw*`).
3. **Artifact Registry**: Deletes all DistillFW container repositories (`distillfw-docker-repo`, `distillfw-repo`).
4. **Vertex AI Prediction Endpoints & Models**: Undeploys and deletes all Vertex AI endpoints and custom models created by DistillFW.
5. **IAM Service Accounts & Bindings**: Deletes all DistillFW IAM service accounts (`distillfw-backend-sa`, `distillfw-trainer-sa`, `distillfw-training-sa`, and any matching `distillfw-*`), and automatically cleans all DistillFW role bindings (including tombstoned `deleted:serviceAccount:distillfw-*` entries) from the project IAM policy.
6. **GCS Workspace Buckets**: Recursively purges and deletes the workspaces bucket (`gs://distillfw-workspaces`) and any additional DistillFW buckets (`gs://distillfw-*`).
7. **Local State & Build Caches**: Removes all Terraform state files (`terraform.tfstate*`, `.terraform/`, `terraform.tfvars`), local filesystem workspaces (`.local_workspace/`), and compiled frontend assets (`frontend/dist/`).

> [!TIP]
> **Provisioning Resiliency**: When running `./deploy.sh` to redeploy, the script automatically checks whether any service accounts, buckets, or repositories already exist in GCP and imports them into Terraform state prior to `terraform apply`. This guarantees that deployments will not fail with `HTTP 409 Conflict: Service account already exists` even if previous runs were interrupted.

---

## 3. Manual Provisioning with Terraform

If your organization manages infrastructure via CI/CD pipelines (e.g., Cloud Build, GitHub Actions), you can provision DistillFW modularly using Terraform.

### 3.1. Configure Terraform Variables
Navigate to `terraform/` and configure your variables:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
project_id             = "your-gcp-project-id"
region                 = "us-central1"
workspaces_bucket_name = "distillfw-workspaces"
backend_image_uri      = "us-central1-docker.pkg.dev/your-gcp-project-id/distillfw-docker-repo/distillfw-backend:latest"
frontend_image_uri     = "us-central1-docker.pkg.dev/your-gcp-project-id/distillfw-docker-repo/distillfw-frontend:latest"
trainer_image_uri      = "us-central1-docker.pkg.dev/your-gcp-project-id/distillfw-docker-repo/distillfw-trainer:latest"
```

### 3.2. Initialize & Apply Terraform
```bash
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### 3.3. Terraform Module Summary
- **[`modules/storage`](file:///usr/local/google/home/raulramos/projects/distillfw/terraform/modules/storage/)**: Creates `distillfw-workspaces` with uniform bucket-level access and CORS rules allowing direct browser streaming of logs and metrics.
- **[`modules/artifact_registry`](file:///usr/local/google/home/raulramos/projects/distillfw/terraform/modules/artifact_registry/)**: Sets up Docker repository `distillfw-docker-repo` for container images.
- **[`modules/iam`](file:///usr/local/google/home/raulramos/projects/distillfw/terraform/modules/iam/)**: Creates `distillfw-backend-sa` (`roles/aiplatform.user`, `roles/storage.admin`, `roles/iam.serviceAccountUser`) and `distillfw-trainer-sa` (`roles/storage.objectAdmin`, `roles/artifactregistry.reader`).
- **[`modules/cloud_run`](file:///usr/local/google/home/raulramos/projects/distillfw/terraform/modules/cloud_run/)**: Provisions Cloud Run v2 services for the backend and Web UI.
- **[`modules/apigee`](file:///usr/local/google/home/raulramos/projects/distillfw/terraform/modules/apigee/)**: Maps `/api/*` to Cloud Run backend and `/*` to Web UI with token/key validation and quota policies.

---

## 4. Building and Pushing Docker Images

### 4.1. Configure Docker Authentication
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 4.2. Build & Push Custom Trainer Image
The custom trainer runs on Vertex AI Custom Training (`nvidia/cuda:12.4.1-runtime-ubuntu22.04` with Python 3.11, PyTorch 2.4, Hugging Face Transformers, PEFT, and BitsAndBytes):

```bash
export REGION="us-central1"
export REPO_URI="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/distillfw-docker-repo"

# Build Trainer
docker build -t "${REPO_URI}/distillfw-trainer:latest" -f trainer/Dockerfile .
docker push "${REPO_URI}/distillfw-trainer:latest"
```

### 4.3. Build & Push Backend API Image
```bash
docker build -t "${REPO_URI}/distillfw-backend:latest" -f backend/Dockerfile .
docker push "${REPO_URI}/distillfw-backend:latest"
```

---

## 5. Local Development Execution

To run DistillFW locally for development, testing, or offline experiments:

### 5.1. Start the Backend API Service
The FastAPI backend automatically detects whether live GCS or local emulation is preferred. In local emulation mode, workspaces are stored under `.local_workspace/`:

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Start backend on port 8080
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### 5.2. Start the Frontend Vite Dev Server
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:3000** in your browser. API requests to `/api/*` will automatically proxy to `http://127.0.0.1:8080`.

### 5.3. Running Automated Tests
DistillFW includes a comprehensive test suite covering status inference, auto-splitting, distillation losses, teacher response normalization, cost calibration probe, and end-to-end API integration:

```bash
python3 -m pytest tests/
```

---

## 6. GCP Quota & Vertex AI Troubleshooting

### 6.1. GPU Quotas in Vertex AI
When launching Vertex AI Custom Jobs with `NVIDIA_L4` or `NVIDIA_A100_80GB`, verify that your GCP project has sufficient quota in your region (e.g. `us-central1`):
1. In Google Cloud Console, navigate to **IAM & Admin > Quotas**.
2. Search for `Custom model training (L4 GPUs per region)` or `Custom model training (A100 GPUs per region)`.
3. If quota is 0, submit an automated quota increase request (typically approved in minutes).

### 6.2. Gemini API Access
Ensure your authenticated account or service account has access to Gemini on Vertex AI. You can verify access via the gcloud CLI:
```bash
gcloud ai models list --region=us-central1
```
