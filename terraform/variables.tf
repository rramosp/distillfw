variable "project_id" {
  description = "Google Cloud Platform Project ID"
  type        = string
}

variable "region" {
  description = "Default GCP Region"
  type        = string
  default     = "us-central1"
}

variable "workspaces_bucket_name" {
  description = "GCS bucket for distillation workspaces"
  type        = string
  default     = "distillfw-workspaces"
}

variable "backend_image_uri" {
  description = "Container image URI for backend Cloud Run service"
  type        = string
  default     = ""
}

variable "frontend_image_uri" {
  description = "Container image URI for frontend Cloud Run service"
  type        = string
  default     = ""
}

variable "trainer_image_uri" {
  description = "Container image URI for custom training Docker container"
  type        = string
  default     = ""
}

variable "allow_public_access" {
  description = "Allow unauthenticated allUsers invoker access on Cloud Run services (set false for domain-restricted enterprise organizations)"
  type        = bool
  default     = false
}

variable "deployer_member" {
  description = "IAM member format (e.g. user:admin@domain.com) to grant invoker access to Cloud Run services"
  type        = string
  default     = ""
}
