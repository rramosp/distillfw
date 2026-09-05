variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "backend_uri" {
  type = string
}

variable "frontend_uri" {
  type = string
}

variable "provision_apigee_org" {
  type    = bool
  default = false
}
