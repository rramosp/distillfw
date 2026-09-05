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
