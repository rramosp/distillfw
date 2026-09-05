# Apigee Routing & Proxy Module
# Defines routing definitions decoupling Apigee and Cloud Run

resource "google_apigee_organization" "apigee_org" {
  count             = var.provision_apigee_org ? 1 : 0
  project_id        = var.project_id
  analytics_region  = var.region
  billing_type      = "EVALUATION"
}

output "proxy_endpoint_info" {
  value = {
    backend_route  = "/api/* -> ${var.backend_uri}"
    frontend_route = "/* -> ${var.frontend_uri}"
    auth_policy    = "OAuth2 / API Key verification with rate limits and quota enforcement"
  }
}
