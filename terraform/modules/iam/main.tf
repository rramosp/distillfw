# Backend Cloud Run Service Account
resource "google_service_account" "backend_sa" {
  account_id   = "distillfw-backend-sa"
  display_name = "DistillFW Backend Service Account"
  project      = var.project_id
}

# Vertex AI Custom Training Service Account
resource "google_service_account" "trainer_sa" {
  account_id   = "distillfw-trainer-sa"
  display_name = "DistillFW Vertex AI Custom Trainer Service Account"
  project      = var.project_id
}

# Role bindings for Backend SA
resource "google_project_iam_member" "backend_aiplatform" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.backend_sa.email}"
}

resource "google_project_iam_member" "backend_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.backend_sa.email}"
}

resource "google_project_iam_member" "backend_actas" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.backend_sa.email}"
}

# Role bindings for Trainer SA
resource "google_project_iam_member" "trainer_storage" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.trainer_sa.email}"
}

resource "google_project_iam_member" "trainer_artifactregistry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.trainer_sa.email}"
}
