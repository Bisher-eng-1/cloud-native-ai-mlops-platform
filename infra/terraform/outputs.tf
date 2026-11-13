output "service_name" {
  description = "Cloud Run v2 service name"
  value       = google_cloud_run_v2_service.inference_api.name
}

output "service_uri" {
  description = "Cloud Run HTTPS endpoint URI once provisioned"
  value       = google_cloud_run_v2_service.inference_api.uri
}
