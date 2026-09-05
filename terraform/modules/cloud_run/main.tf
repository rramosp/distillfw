resource "google_cloud_run_v2_service" "backend" {
  name     = "distillfw-backend"
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = var.backend_sa_email

    containers {
      image = var.backend_image_uri != "" ? var.backend_image_uri : "gcr.io/cloudrun/hello"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "DEFAULT_BUCKET"
        value = var.workspaces_bucket_name
      }
    }
  }
}

resource "google_cloud_run_v2_service" "frontend" {
  name     = "distillfw-frontend"
  location = var.region
  project  = var.project_id
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.frontend_image_uri != "" ? var.frontend_image_uri : "gcr.io/cloudrun/hello"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }
}

# Allow public unauthenticated access (when permitted by organization policies)
resource "google_cloud_run_v2_service_iam_member" "backend_public" {
  count    = var.allow_public_access ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public" {
  count    = var.allow_public_access ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Grant invoker access to the deploying identity
resource "google_cloud_run_v2_service_iam_member" "backend_deployer" {
  count    = var.deployer_member != "" ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = var.deployer_member
}

resource "google_cloud_run_v2_service_iam_member" "frontend_deployer" {
  count    = var.deployer_member != "" ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = var.deployer_member
}
