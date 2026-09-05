variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "backend_sa_email" {
  type = string
}

variable "workspaces_bucket_name" {
  type = string
}

variable "backend_image_uri" {
  type    = string
  default = ""
}

variable "frontend_image_uri" {
  type    = string
  default = ""
}

variable "allow_public_access" {
  description = "Allow unauthenticated allUsers invoker access on Cloud Run services"
  type        = bool
  default     = false
}

variable "deployer_member" {
  description = "IAM member format (e.g. user:admin@domain.com) to grant invoker access"
  type        = string
  default     = ""
}

variable "deletion_protection" {
  description = "Whether to enable deletion protection on Cloud Run services"
  type        = bool
  default     = false
}

