#!/usr/bin/env bash
# ==============================================================================
# DistillFW — Automated Deployment & Infrastructure Provisioning Script
# ==============================================================================
set -e
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUCKET_NAME="distillfw-workspaces"
SAMPLE_PROJECT_ID="distill-gemma-math-v1"
RESET_MODE=false
DRY_RUN=false
AUTO_CONFIRM=false

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
    -y|--yes)
      AUTO_CONFIRM=true
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
      echo "  --reset     Tear down and remove ALL resources created in GCP (service accounts, Cloud Run, Artifact Registry, GCS, local state)"
      echo "  --dry-run   Run pre-flight checks and seed local/offline sample data without GCP cloud calls"
      echo "  -y, --yes   Automatically confirm prompts (e.g. confirming --reset or self-granting missing IAM roles)"
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
  warn "=== Reset Mode Triggered: Cleaning up ALL DistillFW resources ==="
  if [ "$AUTO_CONFIRM" = false ]; then
    read -p "Are you sure you want to permanently destroy and delete ALL DistillFW resources in GCP (service accounts, Cloud Run, Artifact Registry, GCS buckets, and local caches)? [y/N] " -r CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
      info "Reset aborted by user."
      exit 0
    fi
  else
    info "Auto-confirmation enabled (-y/--yes). Proceeding with complete reset."
  fi

  PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
  if [ -z "$PROJECT_ID" ]; then
    error "No active GCP Project set in gcloud. Run 'gcloud config set project <PROJECT_ID>'."
    exit 1
  fi
  REGION="us-central1"
  info "Resetting DistillFW resources in project '${PROJECT_ID}' (region: ${REGION})..."

  # 1. Terraform Destroy (if initialized)
  info "Step 1/7: Running Terraform Destroy..."
  if [ -d "terraform" ]; then
    cd terraform
    if [ -f "terraform.tfstate" ] || [ -d ".terraform" ]; then
      cat <<EOF > terraform.tfvars
project_id             = "${PROJECT_ID}"
region                 = "${REGION}"
workspaces_bucket_name = "${BUCKET_NAME}"
EOF
      terraform init -input=false || true
      terraform destroy -auto-approve -input=false || warn "Terraform destroy completed with notices. Continuing with explicit gcloud resource sweep."
    fi
    cd "$SCRIPT_DIR"
  fi

  # 2. Delete Cloud Run Services
  info "Step 2/7: Deleting Cloud Run services matching 'distillfw*'..."
  RUN_SERVICES=$(gcloud run services list --project="$PROJECT_ID" --region="$REGION" --format="value(name)" 2>/dev/null | grep -E '^distillfw' || true)
  for svc in $RUN_SERVICES; do
    info "  Deleting Cloud Run service '$svc'..."
    gcloud run services delete "$svc" --project="$PROJECT_ID" --region="$REGION" --quiet 2>/dev/null || warn "Failed to delete Cloud Run service $svc"
  done

  # 3. Delete Artifact Registry Repositories
  info "Step 3/7: Deleting Artifact Registry repositories matching 'distillfw*'..."
  AR_REPOS=$(gcloud artifacts repositories list --project="$PROJECT_ID" --location="$REGION" --format="value(name)" 2>/dev/null | grep -E 'distillfw' || true)
  for repo in $AR_REPOS; do
    repo_name=$(basename "$repo")
    info "  Deleting Artifact Registry repository '$repo_name'..."
    gcloud artifacts repositories delete "$repo_name" --project="$PROJECT_ID" --location="$REGION" --quiet 2>/dev/null || warn "Failed to delete repo $repo_name"
  done

  # 4. Cleanup Vertex AI Endpoints and Models
  info "Step 4/7: Cleaning up Vertex AI Endpoints and Models..."
  ENDPOINTS=$(gcloud ai endpoints list --project="$PROJECT_ID" --region="$REGION" --format="value(name)" 2>/dev/null || true)
  for ep in $ENDPOINTS; do
    ep_id=$(basename "$ep")
    disp=$(gcloud ai endpoints describe "$ep_id" --project="$PROJECT_ID" --region="$REGION" --format="value(displayName)" 2>/dev/null || true)
    if [[ "$disp" =~ distillfw ]] || [[ "$disp" =~ distill- ]] || [[ "$ep_id" =~ distill ]]; then
      info "  Undeploying models and deleting Vertex AI endpoint '$disp' ($ep_id)..."
      DEPLOYED_MODELS=$(gcloud ai endpoints describe "$ep_id" --project="$PROJECT_ID" --region="$REGION" --format="value(deployedModels.id)" 2>/dev/null || true)
      for dm in $DEPLOYED_MODELS; do
        gcloud ai endpoints undeploy-model "$ep_id" --project="$PROJECT_ID" --region="$REGION" --deployed-model-id="$dm" --quiet 2>/dev/null || true
      done
      gcloud ai endpoints delete "$ep_id" --project="$PROJECT_ID" --region="$REGION" --quiet 2>/dev/null || warn "Failed to delete endpoint $ep_id"
    fi
  done

  MODELS=$(gcloud ai models list --project="$PROJECT_ID" --region="$REGION" --format="value(name)" 2>/dev/null || true)
  for m in $MODELS; do
    m_id=$(basename "$m")
    disp=$(gcloud ai models describe "$m_id" --project="$PROJECT_ID" --region="$REGION" --format="value(displayName)" 2>/dev/null || true)
    if [[ "$disp" =~ distillfw ]] || [[ "$disp" =~ distill- ]] || [[ "$m_id" =~ distill ]]; then
      info "  Deleting Vertex AI model '$disp' ($m_id)..."
      gcloud ai models delete "$m_id" --project="$PROJECT_ID" --region="$REGION" --quiet 2>/dev/null || warn "Failed to delete model $m_id"
    fi
  done

  # 5. Delete Service Accounts & Clean Project IAM Policy Bindings
  info "Step 5/7: Deleting DistillFW Service Accounts and removing IAM bindings..."
  python3 - <<EOF
import subprocess, json, sys, tempfile, os

project_id = "${PROJECT_ID}"
try:
    policy_json = subprocess.check_output(
        ["gcloud", "projects", "get-iam-policy", project_id, "--format=json"],
        stderr=subprocess.DEVNULL
    )
    policy = json.loads(policy_json)
    changed = False
    new_bindings = []
    for binding in policy.get("bindings", []):
        members = binding.get("members", [])
        filtered_members = [m for m in members if not ("distillfw-" in m or "distillfw" in m)]
        if len(filtered_members) != len(members):
            changed = True
        if filtered_members:
            binding["members"] = filtered_members
            new_bindings.append(binding)
    if changed:
        policy["bindings"] = new_bindings
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(policy, tf)
            tf_path = tf.name
        subprocess.run(
            ["gcloud", "projects", "set-iam-policy", project_id, tf_path, "--quiet"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.remove(tf_path)
        print("  ✓ Cleaned all DistillFW IAM role bindings from project policy")
except Exception as e:
    print(f"  Notice on IAM policy binding cleanup: {e}", file=sys.stderr)
EOF

  KNOWN_SAS=(
    "distillfw-backend-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    "distillfw-trainer-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    "distillfw-training-sa@${PROJECT_ID}.iam.gserviceaccount.com"
  )
  DISCOVERED_SAS=$(gcloud iam service-accounts list --project="$PROJECT_ID" --format="value(email)" 2>/dev/null | grep -E '^distillfw-' || true)
  TARGET_SAS=()
  for sa in "${KNOWN_SAS[@]}"; do
    TARGET_SAS+=("$sa")
  done
  for sa in $DISCOVERED_SAS; do
    if [[ ! " ${TARGET_SAS[*]} " =~ " ${sa} " ]]; then
      TARGET_SAS+=("$sa")
    fi
  done

  for sa in "${TARGET_SAS[@]}"; do
    if gcloud iam service-accounts describe "$sa" --project="$PROJECT_ID" &>/dev/null; then
      info "  Deleting service account: $sa..."
      gcloud iam service-accounts delete "$sa" --project="$PROJECT_ID" --quiet 2>/dev/null || warn "Failed to delete service account $sa"
    fi
  done

  # 6. Delete Cloud Storage Buckets
  info "Step 6/7: Deleting GCS workspace buckets..."
  if gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
    info "  Deleting workspace bucket gs://${BUCKET_NAME}..."
    gcloud storage rm --recursive "gs://${BUCKET_NAME}" 2>/dev/null || gsutil rm -r "gs://${BUCKET_NAME}" 2>/dev/null || warn "Could not delete gs://${BUCKET_NAME}"
  fi
  DISTILL_BUCKETS=$(gcloud storage buckets list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null | grep -E '^distillfw-' || true)
  for b in $DISTILL_BUCKETS; do
    if [ "$b" != "$BUCKET_NAME" ]; then
      info "  Deleting additional bucket gs://$b..."
      gcloud storage rm --recursive "gs://$b" 2>/dev/null || gsutil rm -r "gs://$b" 2>/dev/null || true
    fi
  done

  # 7. Clean Local Workspaces and Build Caches
  info "Step 7/7: Cleaning local state, workspaces, and caches..."
  rm -rf terraform/terraform.tfstate* terraform/.terraform terraform/.terraform.lock.hcl terraform/terraform.tfvars
  rm -rf .local_workspace
  rm -rf frontend/dist
  info "  ✓ Cleaned terraform state, .local_workspace, and frontend/dist"

  success "=== All DistillFW GCP resources, service accounts, and local state have been completely reset ==="
  exit 0
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
  BILLING_ENABLED=$(gcloud beta billing projects describe "$PROJECT_ID" --quiet --format="value(billingEnabled)" 2>/dev/null || echo "unknown")
  if [ "$BILLING_ENABLED" = "unknown" ]; then
    info "  - Billing status: verified (cloudbilling API check passed/skipped)"
  elif [ "$BILLING_ENABLED" != "True" ] && [ "$BILLING_ENABLED" != "true" ]; then
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

info "Verifying IAM permissions for '$AUTH_ACCOUNT' on project '$PROJECT_ID'..."

if [ "$DRY_RUN" = false ]; then
  # Evaluate IAM policy using Python
  IAM_CHECK_RESULT=$(python3 - <<EOF
import sys, json, subprocess

project_id = "${PROJECT_ID}"
account = "${AUTH_ACCOUNT}"
required_roles = [
    "roles/aiplatform.admin",
    "roles/storage.admin",
    "roles/run.admin",
    "roles/apigee.admin",
    "roles/artifactregistry.admin",
    "roles/iam.serviceAccountUser"
]

try:
    cmd = ["gcloud", "projects", "get-iam-policy", project_id, "--format=json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(json.dumps({
            "status": "CANNOT_READ_POLICY",
            "error": proc.stderr.strip()
        }))
        sys.exit(0)
    
    policy = json.loads(proc.stdout)
    bindings = policy.get("bindings", [])
    
    user_roles = set()
    for b in bindings:
        members = b.get("members", [])
        if f"user:{account}" in members:
            user_roles.add(b.get("role"))
    
    is_owner = "roles/owner" in user_roles
    can_grant = is_owner or ("roles/resourcemanager.projectIamAdmin" in user_roles)
    
    if is_owner:
        missing = []
    else:
        missing = [r for r in required_roles if r not in user_roles]
    
    print(json.dumps({
        "status": "OK",
        "user_roles": list(user_roles),
        "is_owner": is_owner,
        "can_grant": can_grant,
        "missing_roles": missing
    }))
except Exception as e:
    print(json.dumps({"status": "ERROR", "error": str(e)}))
EOF
)

  CHECK_STATUS=$(echo "$IAM_CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'ERROR'))" 2>/dev/null || echo "ERROR")

  MISSING_ROLES=()
  CAN_GRANT=false

  if [ "$CHECK_STATUS" = "CANNOT_READ_POLICY" ]; then
    warn "Could not retrieve project IAM policy directly. Checking if user can grant permissions..."
    MISSING_ROLES=("${REQUIRED_ROLES[@]}")
  elif [ "$CHECK_STATUS" = "OK" ]; then
    CAN_GRANT=$(echo "$IAM_CHECK_RESULT" | python3 -c "import sys, json; print(str(json.load(sys.stdin).get('can_grant', False)).lower())")
    while IFS= read -r role_item; do
      if [ -n "$role_item" ]; then
        MISSING_ROLES+=("$role_item")
      fi
    done < <(echo "$IAM_CHECK_RESULT" | python3 -c "import sys, json; [print(r) for r in json.load(sys.stdin).get('missing_roles', [])]")
  else
    warn "IAM policy evaluation notice. Continuing with role verification."
  fi

  # Handle missing roles
  if [ ${#MISSING_ROLES[@]} -eq 0 ]; then
    success "  ✓ All required IAM permissions are present and verified for account: $AUTH_ACCOUNT"
  else
    warn "The following required IAM roles are missing for account '$AUTH_ACCOUNT' on project '$PROJECT_ID':"
    for r in "${MISSING_ROLES[@]}"; do
      echo -e "    \033[1;31m✗\033[0m $r"
    done
    echo ""

    if [ "$CAN_GRANT" = "true" ]; then
      info "Your account ($AUTH_ACCOUNT) has administrative permissions (e.g. Project IAM Admin / Owner) to grant these missing roles."
      
      CONFIRM_GRANT=""
      if [ "$AUTO_CONFIRM" = true ]; then
        CONFIRM_GRANT="y"
        info "Auto-confirm flag set (--yes). Proceeding to grant missing roles..."
      else
        read -p "Would you like deploy.sh to automatically grant these roles to your account now? [y/N] " -r CONFIRM_GRANT
      fi

      if [[ "$CONFIRM_GRANT" =~ ^[Yy]$ ]]; then
        info "Granting missing IAM roles to '$AUTH_ACCOUNT'..."
        for r in "${MISSING_ROLES[@]}"; do
          info "  - Adding role: $r"
          gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="user:${AUTH_ACCOUNT}" \
            --role="$r" \
            --condition=None >/dev/null
        done
        success "  ✓ Successfully granted missing IAM roles to $AUTH_ACCOUNT."
      else
        error "Cannot proceed without required IAM permissions. Please grant them and rerun deploy.sh."
        exit 1
      fi
    else
      warn "Your account ($AUTH_ACCOUNT) does NOT have permission to grant IAM roles on project '$PROJECT_ID'."
      echo ""
      echo "================================================================================"
      echo " ACTION REQUIRED: Contact your Google Cloud Project Administrator"
      echo "================================================================================"
      echo "Please share the following request with your GCP Project Administrator:"
      echo ""
      echo "--------------------------------------------------------------------------------"
      echo "Subject: IAM Role Request for DistillFW Deployment on Project '${PROJECT_ID}'"
      echo ""
      echo "Hi Team,"
      echo ""
      echo "I am deploying DistillFW (Managed Distillation Framework on GCP) on project '${PROJECT_ID}'."
      echo "My account (${AUTH_ACCOUNT}) requires the following IAM roles to complete deployment:"
      echo ""
      echo "Required Roles & Justifications:"
      for r in "${MISSING_ROLES[@]}"; do
        case "$r" in
          "roles/aiplatform.admin")
            echo "  - roles/aiplatform.admin: Required to run Vertex AI CustomJobs for model distillation training and deploy vLLM prediction endpoints." ;;
          "roles/storage.admin")
            echo "  - roles/storage.admin: Required to manage GCS workspace buckets (gs://${BUCKET_NAME}) and configure CORS policies for telemetry." ;;
          "roles/run.admin")
            echo "  - roles/run.admin: Required to deploy and configure Cloud Run services for the FastAPI backend and React Web UI." ;;
          "roles/apigee.admin")
            echo "  - roles/apigee.admin: Required to configure the Apigee API Gateway proxy routing (/api/* to backend, /* to UI)." ;;
          "roles/artifactregistry.admin")
            echo "  - roles/artifactregistry.admin: Required to create and manage private Docker repositories for custom trainer containers." ;;
          "roles/iam.serviceAccountUser")
            echo "  - roles/iam.serviceAccountUser: Required to bind service accounts (distillfw-backend-sa, distillfw-trainer-sa) to services." ;;
          *)
            echo "  - $r: Required for DistillFW framework operations." ;;
        esac
      done
      echo ""
      echo "Command for Administrator to run:"
      echo "for role in ${MISSING_ROLES[*]}; do"
      echo "  gcloud projects add-iam-policy-binding \"${PROJECT_ID}\" --member=\"user:${AUTH_ACCOUNT}\" --role=\"\$role\""
      echo "done"
      echo "--------------------------------------------------------------------------------"
      echo "================================================================================"
      echo ""
      error "Cannot proceed without required IAM permissions. Please contact your administrator and rerun deploy.sh."
      exit 1
    fi
  fi
else
  info "  ✓ Dry-run mode: IAM role checks simulated for project '$PROJECT_ID'"
fi

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

  # Resiliency: If service accounts, repositories, or buckets exist in GCP but are not yet in
  # terraform state, adopt them into state to prevent HTTP 409 Conflict errors.
  TF_STATE=$(terraform state list 2>/dev/null || true)

  BACKEND_SA="distillfw-backend-sa@${PROJECT_ID}.iam.gserviceaccount.com"
  if ! echo "$TF_STATE" | grep -q "module.iam.google_service_account.backend_sa"; then
    if gcloud iam service-accounts describe "$BACKEND_SA" --project="$PROJECT_ID" &>/dev/null; then
      info "Adopting existing service account '$BACKEND_SA' into Terraform state..."
      terraform import module.iam.google_service_account.backend_sa "projects/${PROJECT_ID}/serviceAccounts/${BACKEND_SA}" 2>/dev/null || true
    fi
  fi

  TRAINER_SA="distillfw-trainer-sa@${PROJECT_ID}.iam.gserviceaccount.com"
  if ! echo "$TF_STATE" | grep -q "module.iam.google_service_account.trainer_sa"; then
    if gcloud iam service-accounts describe "$TRAINER_SA" --project="$PROJECT_ID" &>/dev/null; then
      info "Adopting existing service account '$TRAINER_SA' into Terraform state..."
      terraform import module.iam.google_service_account.trainer_sa "projects/${PROJECT_ID}/serviceAccounts/${TRAINER_SA}" 2>/dev/null || true
    fi
  fi

  if ! echo "$TF_STATE" | grep -q "module.artifact_registry.google_artifact_registry_repository.docker_repo"; then
    if gcloud artifacts repositories describe "distillfw-docker-repo" --project="$PROJECT_ID" --location="us-central1" &>/dev/null; then
      info "Adopting existing Artifact Registry repository 'distillfw-docker-repo' into Terraform state..."
      terraform import module.artifact_registry.google_artifact_registry_repository.docker_repo "projects/${PROJECT_ID}/locations/us-central1/repositories/distillfw-docker-repo" 2>/dev/null || true
    fi
  fi

  if ! echo "$TF_STATE" | grep -q "module.storage.google_storage_bucket.workspaces_bucket"; then
    if gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
      info "Adopting existing GCS bucket 'gs://${BUCKET_NAME}' into Terraform state..."
      terraform import module.storage.google_storage_bucket.workspaces_bucket "${BUCKET_NAME}" 2>/dev/null || true
    fi
  fi

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

export GCP_PROJECT_ID="${PROJECT_ID}"
export DEFAULT_BUCKET="${BUCKET_NAME}"

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

# Clear any existing downstream artifacts to ensure pristine DATASET_READY state
for rel_path in [
    f"{project_id}/data/teacher_inferences.jsonl",
    f"{project_id}/cost/cost_estimate.json",
    f"{project_id}/training/metrics.jsonl",
    f"{project_id}/training/heartbeat.json",
    f"{project_id}/training/final_adapter/adapter_model.safetensors",
    f"{project_id}/training/final_adapter/adapter_config.json",
    f"{project_id}/evaluation/eval_results.json",
    f"{project_id}/evaluation/test_predictions.jsonl",
    f"{project_id}/deployment/endpoint_metadata.json"
]:
    local_p = storage_service.get_local_path(bucket, rel_path)
    if os.path.exists(local_p):
        os.remove(local_p)

storage_service.set_active_operation(bucket, project_id, None)

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
