provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Storage Module
module "storage" {
  source      = "./modules/storage"
  project_id  = var.project_id
  region      = var.region
  bucket_name = var.workspaces_bucket_name
}

# 2. Artifact Registry Module
module "artifact_registry" {
  source     = "./modules/artifact_registry"
  project_id = var.project_id
  region     = var.region
}

# 3. IAM Module
module "iam" {
  source     = "./modules/iam"
  project_id = var.project_id
}

# 4. Cloud Run Module
module "cloud_run" {
  source                 = "./modules/cloud_run"
  project_id             = var.project_id
  region                 = var.region
  backend_sa_email       = module.iam.backend_sa_email
  workspaces_bucket_name = module.storage.bucket_name
  backend_image_uri      = var.backend_image_uri
  frontend_image_uri     = var.frontend_image_uri
  allow_public_access    = var.allow_public_access
  deployer_member        = var.deployer_member
}

# 5. Apigee API Gateway Module
module "apigee" {
  source       = "./modules/apigee"
  project_id   = var.project_id
  region       = var.region
  backend_uri  = module.cloud_run.backend_uri
  frontend_uri = module.cloud_run.frontend_uri
}
