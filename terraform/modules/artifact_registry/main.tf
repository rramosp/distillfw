resource "google_artifact_registry_repository" "docker_repo" {
  provider      = google
  project       = var.project_id
  location      = var.region
  repository_id = "distillfw-docker-repo"
  description   = "Docker repository for DistillFW training and API images"
  format        = "DOCKER"
}
