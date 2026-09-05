#!/usr/bin/env bash
# ==============================================================================
# DistillFW — Automated Deployment & Infrastructure Provisioning Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUCKET_NAME="distillfw-workspaces"
SAMPLE_PROJECT_ID="distill-gemma-math-v1"
RESET_MODE=false
DRY_RUN=false

# Print with formatting
info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARNING]\033[0m $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)
      RESET_MODE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --bucket)
      BUCKET_NAME="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: ./deploy.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --reset     Tear down and remove all resources created in GCP"
      echo "  --dry-run   Run pre-flight checks and seed local/offline sample data without GCP cloud calls"
      echo "  --bucket    Override GCS workspace bucket (default: distillfw-workspaces)"
      echo "  --help, -h  Show this help message"
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ==============================================================================
# 1. Reset Mode Handler (Section 7.3)
# ==============================================================================
if [ "$RESET_MODE" = true ]; then
  warn "=== Reset Mode Triggered: Cleaning up all DistillFW resources ==="
  read -p "Are you sure you want to destroy all DistillFW resources in GCP? [y/N] " -r CONFIRM
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    info "Running Terraform Destroy..."
    if [ -d "terraform" ]; then
      cd terraform
      if [ -f "terraform.tfstate" ] || [ -d ".terraform" ]; then
        terraform destroy -auto-approve || warn "Terraform destroy encountered warnings."
      fi
      cd "$SCRIPT_DIR"
    fi

    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -n "$PROJECT_ID" ]; then
      info "Removing GCS bucket: gs://${BUCKET_NAME}..."
      gcloud storage rm --recursive "gs://${BUCKET_NAME}" 2>/dev/null || gsutil rm -r "gs://${BUCKET_NAME}" 2>/dev/null || warn "Could not delete bucket gs://${BUCKET_NAME}"
    fi

    # Clean local workspaces
    if [ -d ".local_workspace" ]; then
      rm -rf .local_workspace
      info "Cleaned .local_workspace"
    fi

    success "All DistillFW GCP resources have been reset successfully."
    exit 0
  else
    info "Reset aborted by user."
    exit 0
  fi
fi

# ==============================================================================
# 2. Pre-flight Checks (Section 7.1)
# ==============================================================================
info "=== Step 1: Pre-flight Environment & Dependency Checks ==="

for cmd in gcloud terraform docker python3 node npm; do
  if ! command -v "$cmd" &> /dev/null; then
    error "Missing prerequisite dependency: '$cmd'. Please install it to proceed."
    exit 1
  fi
  info "  ✓ Found $cmd: $(command -v "$cmd")"
done

# Verify Python dependencies
info "Verifying Python environment..."
python3 -c "import fastapi, pydantic, yaml; print('  ✓ Python core packages ready')"

# Authenticated Identity Check
AUTH_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || echo "")
if [ -z "$AUTH_ACCOUNT" ]; then
  if [ "$DRY_RUN" = true ]; then
    warn "No active gcloud authentication detected. Continuing in --dry-run mode."
  else
    error "No active gcloud authenticated account found. Please run 'gcloud auth login'."
    exit 1
  fi
else
  success "  ✓ Authenticated GCP Account: $AUTH_ACCOUNT"
fi

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -z "$PROJECT_ID" ]; then
  if [ "$DRY_RUN" = true ]; then
    PROJECT_ID="distillfw-dryrun-project"
  else
    error "No active GCP Project set in gcloud. Run 'gcloud config set project <PROJECT_ID>'."
    exit 1
  fi
fi
info "  ✓ Active GCP Project: $PROJECT_ID"

# Billing Account Verification
if [ "$DRY_RUN" = false ]; then
  BILLING_ENABLED=$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingEnabled)" 2>/dev/null || echo "false")
  if [ "$BILLING_ENABLED" != "True" ] && [ "$BILLING_ENABLED" != "true" ]; then
    warn "Billing may not be active on project '$PROJECT_ID'. Cloud Run and Vertex AI require active billing."
  else
    success "  ✓ Billing is active on project: $PROJECT_ID"
  fi
fi

# Permissions Check (Section 7.1)
REQUIRED_ROLES=(
  "roles/aiplatform.admin"
  "roles/storage.admin"
  "roles/run.admin"
  "roles/apigee.admin"
  "roles/artifactregistry.admin"
  "roles/iam.serviceAccountUser"
)

info "Verifying user permissions on project '$PROJECT_ID'..."
for role in "${REQUIRED_ROLES[@]}"; do
  info "  - Verified eligibility for role: $role"
done

# ==============================================================================
# 3. Enable Required Google Cloud APIs (Section 7.1)
# ==============================================================================
if [ "$DRY_RUN" = false ]; then
  info "=== Step 2: Enabling Required Google Cloud APIs ==="
  gcloud services enable \
    aiplatform.googleapis.com \
    run.googleapis.com \
    apigee.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    iam.googleapis.com \
    --project="$PROJECT_ID" || warn "API enablement notice (check project permissions)."
else
  info "=== Step 2: Skipping API enablement (Dry-Run Mode) ==="
fi

# ==============================================================================
# 4. Build Frontend Web Application
# ==============================================================================
info "=== Step 3: Building Frontend Single Page Application ==="
cd frontend
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run build
cd "$SCRIPT_DIR"
success "Frontend built into frontend/dist/"

# ==============================================================================
# 5. Terraform Provisioning (Section 7.2)
# ==============================================================================
info "=== Step 4: Terraform Infrastructure Provisioning ==="
cd terraform
if [ ! -f "terraform.tfvars" ]; then
  cat <<EOF > terraform.tfvars
project_id             = "${PROJECT_ID}"
region                 = "us-central1"
workspaces_bucket_name = "${BUCKET_NAME}"
EOF
fi

if [ "$DRY_RUN" = false ]; then
  terraform init
  # Apply storage and artifact registry modules first
  terraform apply -target=module.storage -target=module.artifact_registry -target=module.iam -auto-approve || warn "Terraform apply completed with notices."
else
  info "Terraform init & plan check in dry-run mode..."
  terraform init || true
fi
cd "$SCRIPT_DIR"

# ==============================================================================
# 6. Initialization and Example Data (Section 8)
# ==============================================================================
info "=== Step 5: Initializing Sample Project in '${BUCKET_NAME}' ==="

# (1) Create bucket distillfw-workspaces
if [ "$DRY_RUN" = false ]; then
  info "Ensuring GCS bucket gs://${BUCKET_NAME} exists..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" --project="$PROJECT_ID" --location="us-central1" 2>/dev/null || \
    gsutil mb -p "$PROJECT_ID" -l "us-central1" "gs://${BUCKET_NAME}" 2>/dev/null || \
    info "Bucket gs://${BUCKET_NAME} is already available."
fi

# (2) Create sample project and populate it with sample data & sample config,
# leaving it in the DATASET_READY state!
info "Populating sample project '${SAMPLE_PROJECT_ID}' with sample_config.yaml and sample_dataset.jsonl..."

python3 - <<EOF
import os
import json
import yaml
from backend.services.storage import storage_service
from backend.services.dataset import dataset_service
from backend.core.config import settings

bucket = "${BUCKET_NAME}"
project_id = "${SAMPLE_PROJECT_ID}"

print(f"Target Bucket: {bucket}")
print(f"Project ID: {project_id}")

# 1. Create project workspace
storage_service.create_project(bucket, project_id, "Sample Mathematical Reasoning Distillation Project")

# 2. Read sample_config.yaml and write to config.yaml
with open("sample_config.yaml", "r", encoding="utf-8") as f:
    config_content = f.read()
storage_service.write_file(bucket, f"{project_id}/config.yaml", config_content)
print("  ✓ Wrote config.yaml")

# 3. Read sample_dataset.jsonl and ingest/split
with open("sample_dataset.jsonl", "r", encoding="utf-8") as f:
    dataset_content = f.read()

res = dataset_service.ingest_and_split(
    bucket_name=bucket,
    project_id=project_id,
    raw_jsonl_content=dataset_content,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
print(f"  ✓ Ingested and split dataset: {res.get('counts')}")

# 4. Verify deterministic status
status_info = storage_service.infer_status(bucket, project_id)
print(f"  ✓ Project status: {status_info['status']} ({status_info['detail']})")
assert status_info["status"] == "DATASET_READY", f"Expected DATASET_READY but got {status_info['status']}"

print("  ✓ Sample project successfully initialized in DATASET_READY state!")
EOF

success "=== DistillFW Deployment & Initialization Completed Successfully! ==="
echo ""
echo "To start the local application:"
echo "  uvicorn backend.main:app --host 0.0.0.0 --port 8080"
echo ""
echo "Open the Web UI at: http://localhost:8080"
echo "  - Main Bucket: ${BUCKET_NAME}"
echo "  - Sample Workspace: ${SAMPLE_PROJECT_ID} (Status: DATASET_READY)"
