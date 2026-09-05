output "workspaces_bucket" {
  description = "GCS bucket for workspaces"
  value       = module.storage.bucket_name
}

output "artifact_registry_repo" {
  description = "Artifact Registry Docker repository"
  value       = module.artifact_registry.repository_id
}

output "backend_service_uri" {
  description = "Cloud Run backend service URL"
  value       = module.cloud_run.backend_uri
}

output "frontend_service_uri" {
  description = "Cloud Run frontend service URL"
  value       = module.cloud_run.frontend_uri
}

output "backend_service_account" {
  description = "Service account used by Cloud Run backend"
  value       = module.iam.backend_sa_email
}

output "trainer_service_account" {
  description = "Service account used by Vertex AI custom training"
  value       = module.iam.trainer_sa_email
}
