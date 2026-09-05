output "backend_sa_email" {
  value = google_service_account.backend_sa.email
}

output "trainer_sa_email" {
  value = google_service_account.trainer_sa.email
}
